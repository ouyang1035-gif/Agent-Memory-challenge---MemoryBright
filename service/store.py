"""存储层：Qdrant（向量）+ SQLite（元数据），负责初始化/写入/检索/幂等去重。

- Qdrant 单 collection，cosine 距离，payload 存 user_id/session_id/request_id/role/created_at
- SQLite 表 memories，request_id 建索引用于幂等去重
- 写入流程：先查 SQLite 幂等去重 -> embedding -> 写 SQLite -> 写 Qdrant -> 返回
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

import config

logger = logging.getLogger(__name__)

_qdrant: QdrantClient | None = None
_db_lock = threading.Lock()
# qdrant-client 的 local-file 模式内部用 sqlite 存 payload，并发调用不是线程安全的
# （多线程同时 upsert/query 会抛 InterfaceError / OperationalError: database is locked）。
# store 的调用已移入线程池，这里必须把所有 Qdrant 访问串行化。
_q_lock = threading.RLock()

# ---------- 内存向量缓存 ----------
# 本地 Qdrant（local-file 模式）不支持 payload 索引，带 user_id 过滤的检索会
# 逐点扫描整个 collection：18k 点约 2s，且随写入线性变慢（线上"一次比一次慢"
# 的主因之一，MCQ 多选项时一次 /search 还会放大 5~11 倍）。这里把全量向量
# 常驻内存（float32，18k×1024 ≈ 74MB），按 user 取子集做 cosine，单次 ~1ms。
_vc_lock = threading.Lock()
_vc_ready = False
_vc_ids: list[str] = []
_vc_pos: dict[str, int] = {}
_vc_users: dict[str, list[int]] = {}
_vc_mat: np.ndarray | None = None
_vc_count = 0


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _coerce_str(value) -> str:
    """把任意值安全转成 str，供 SQLite 绑定（避免 dict/list 触发 ProgrammingError）。"""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def init_store() -> None:
    """应用启动时调用：初始化 Qdrant collection 与 SQLite 表，并做 schema 迁移。"""
    global _qdrant

    # SQLite 初始化
    with _db_lock, _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # schema 迁移：旧表无新列则 ALTER TABLE ADD COLUMN，不丢已有数据
        existing_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        new_columns = {
            "type": "TEXT NOT NULL DEFAULT 'other'",
            "entities": "TEXT NOT NULL DEFAULT '[]'",
            "time_expr": "TEXT",
            "is_correction": "INTEGER NOT NULL DEFAULT 0",
            "superseded": "INTEGER NOT NULL DEFAULT 0",
            "importance": "REAL NOT NULL DEFAULT 1.0",
            # P1：双表示。kind='chunk' 为原始分块，kind='fact' 为原子事实；
            # 旧数据无该列时默认补 'chunk'（向后兼容）。
            "kind": "TEXT NOT NULL DEFAULT 'chunk'",
            # 事实所属分块的 memory_id（分块自身为 NULL）
            "source_id": "TEXT",
        }
        for col_name, col_def in new_columns.items():
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")
                logger.info("migrated: added column %s to memories", col_name)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_request_id "
            "ON memories(request_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_superseded "
            "ON memories(user_id, superseded)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_kind "
            "ON memories(user_id, kind)"
        )
        # P2：实体关联倒排索引（entity -> memory_id，按 user 隔离）
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_index (
                entity TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                user_id TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_index_user_entity "
            "ON entity_index(user_id, entity)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_index_unique "
            "ON entity_index(user_id, entity, memory_id)"
        )
        conn.commit()
    logger.info("SQLite initialized at %s", config.SQLITE_DB_PATH)

    # Qdrant 初始化
    # 优先本地文件模式（无需 Docker，适合网络受限环境）；
    # 若配置了 QDRANT_HOST 且不是 "local"，则连接远程/Docker Qdrant 服务。
    if config.QDRANT_HOST.lower() in ("local", "", ":memory:"):
        # 本地文件模式：数据持久化到 ./qdrant_storage 目录
        _qdrant = QdrantClient(path=config.QDRANT_LOCAL_PATH)
        logger.info("Qdrant running in local-file mode at %s", config.QDRANT_LOCAL_PATH)
    else:
        _qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        logger.info(
            "Qdrant connecting to %s:%d", config.QDRANT_HOST, config.QDRANT_PORT
        )

    existing = {c.name for c in _qdrant.get_collections().collections}
    if config.QDRANT_COLLECTION not in existing:
        _qdrant.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM, distance=Distance.COSINE
            ),
        )
        logger.info("Qdrant collection '%s' created", config.QDRANT_COLLECTION)
    else:
        logger.info("Qdrant collection '%s' already exists", config.QDRANT_COLLECTION)

    # 后台预热内存向量缓存（未就绪时检索自动回落 Qdrant 过滤查询）
    start_vector_cache()


def close_store() -> None:
    """应用关闭时释放资源。"""
    global _qdrant
    if _qdrant is not None:
        _qdrant.close()
        _qdrant = None


# ---------- 内存向量缓存实现 ----------


def _vc_normalize(vectors) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _vc_ensure_capacity(extra: int) -> None:
    """在 _vc_lock 内调用；按需扩容（翻倍）并保留已有行。"""
    global _vc_mat
    need = _vc_count + extra
    cap = 0 if _vc_mat is None else _vc_mat.shape[0]
    if need <= cap:
        return
    new_cap = max(need, int(cap * 1.25) + 64, 1024)
    dim = None if _vc_mat is None else _vc_mat.shape[1]
    if dim is None:
        return  # 首次由 _vc_reset 分配
    mat = np.zeros((new_cap, dim), dtype=np.float32)
    mat[:_vc_count] = _vc_mat[:_vc_count]
    _vc_mat = mat


def _vc_reset(vectors, ids: list[str], user_ids: list[str]) -> None:
    """在 _vc_lock 内调用：用一批数据重建缓存。"""
    global _vc_mat, _vc_count, _vc_ids, _vc_pos, _vc_users
    arr = _vc_normalize(vectors) if len(vectors) else None
    _vc_count = 0 if arr is None else arr.shape[0]
    _vc_mat = None if arr is None else np.zeros(
        (max(int(_vc_count * 1.25) + 64, 1024), arr.shape[1]), dtype=np.float32
    )
    if arr is not None:
        _vc_mat[:_vc_count] = arr
    _vc_ids = []
    _vc_pos = {}
    _vc_users = {}
    for i, mid in enumerate(ids):
        _vc_ids.append(mid)
        _vc_pos[mid] = i
        _vc_users.setdefault(user_ids[i], []).append(i)


def _vc_append(ids: list[str], vectors, user_ids: list[str]) -> None:
    """追加新写入的点（/add 后立即生效，保证写后可检索）。"""
    global _vc_count
    if not ids or not _vc_ready or _vc_mat is None:
        return
    arr = _vc_normalize(vectors)
    if arr.shape[1] != _vc_mat.shape[1]:
        logger.warning(
            "vector cache dim mismatch: %d != %d; skip append",
            arr.shape[1],
            _vc_mat.shape[1],
        )
        return
    with _vc_lock:
        _vc_ensure_capacity(len(ids))
        for i, mid in enumerate(ids):
            pos = _vc_pos.get(mid)
            if pos is None:
                pos = _vc_count
                _vc_mat[pos] = arr[i]
                _vc_count += 1
                _vc_pos[mid] = pos
                _vc_ids.append(mid)
                _vc_users.setdefault(user_ids[i], []).append(pos)
            else:
                _vc_mat[pos] = arr[i]


def _vc_search(user_id: str, query_vector: list[float], top_k: int):
    """缓存就绪时按 user 子集做 cosine 排序；未就绪返回 None（由调用方回落 Qdrant）。"""
    if not _vc_ready or _vc_mat is None:
        return None
    q = _vc_normalize(query_vector)[0]
    with _vc_lock:
        rows = _vc_users.get(user_id)
        if not rows:
            return []
        idx = np.asarray(rows, dtype=np.int64)
        sims = _vc_mat[idx] @ q
        ids = [ _vc_ids[int(i)] for i in idx ]
    k = min(top_k, len(rows))
    if k < len(rows):
        part = np.argpartition(-sims, k - 1)[:k]
        order = part[np.argsort(-sims[part])]
    else:
        order = np.argsort(-sims)
    return [(ids[int(j)], float(sims[int(j)])) for j in order]


def _scroll_points(with_vectors: bool):
    client = _ensure_qdrant()
    offset = None
    while True:
        with _q_lock:
            points, offset = client.scroll(
                collection_name=config.QDRANT_COLLECTION,
                limit=512,
                offset=offset,
                with_payload=["user_id"],
                with_vectors=with_vectors,
            )
        for point in points:
            yield point
        if offset is None:
            break


def _point_vector(point):
    vec = point.vector
    if isinstance(vec, dict):
        vec = next(iter(vec.values()), None)
    return vec


def _build_vector_cache() -> None:
    global _vc_ready
    if not getattr(config, "ENABLE_VECTOR_CACHE", True):
        return
    t0 = time.perf_counter()
    try:
        ids: list[str] = []
        vecs: list[list[float]] = []
        users: list[str] = []
        for point in _scroll_points(with_vectors=True):
            vec = _point_vector(point)
            if not vec:
                continue
            ids.append(str(point.id))
            vecs.append(vec)
            users.append(str((point.payload or {}).get("user_id") or ""))
        with _vc_lock:
            _vc_reset(vecs, ids, users)
        _vc_ready = True
        logger.info(
            "vector cache ready: %d points in %.2fs", len(ids),
            time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("vector cache build failed: %s", type(exc).__name__)
        return

    # 构建期间可能有 /add 写入（此时缓存未就绪，appended 被跳过），补一次增量同步
    try:
        missing: list[tuple[str, str]] = []
        for point in _scroll_points(with_vectors=False):
            mid = str(point.id)
            with _vc_lock:
                known = mid in _vc_pos
            if not known:
                missing.append((mid, str((point.payload or {}).get("user_id") or "")))
        if missing:
            client = _ensure_qdrant()
            for i in range(0, len(missing), 256):
                batch = missing[i : i + 256]
                with _q_lock:
                    records = client.retrieve(
                        collection_name=config.QDRANT_COLLECTION,
                        ids=[mid for mid, _ in batch],
                        with_vectors=True,
                    )
                user_map = dict(batch)
                got_ids: list[str] = []
                got_vecs: list[list[float]] = []
                got_users: list[str] = []
                for rec in records:
                    vec = _point_vector(rec)
                    if not vec:
                        continue
                    got_ids.append(str(rec.id))
                    got_vecs.append(vec)
                    got_users.append(user_map.get(str(rec.id), ""))
                _vc_append(got_ids, got_vecs, got_users)
            logger.info("vector cache incremental sync: +%d points", len(missing))
    except Exception as exc:  # noqa: BLE001
        logger.warning("vector cache sync failed: %s", type(exc).__name__)


def start_vector_cache() -> None:
    """后台构建向量缓存，避免拖慢 8000 端口启动。"""
    if not getattr(config, "ENABLE_VECTOR_CACHE", True):
        logger.info("vector cache disabled by config")
        return
    threading.Thread(target=_build_vector_cache, name="vec-cache", daemon=True).start()


def _ensure_qdrant() -> QdrantClient:
    if _qdrant is None:
        raise RuntimeError("store not initialized")
    return _qdrant


def request_exists(request_id: str) -> bool:
    """幂等检查：该 request_id 是否已成功写入过。"""
    with _db_lock, _get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM memories WHERE request_id = ? LIMIT 1", (request_id,)
        ).fetchone()
    return row is not None


def _iso_from_ms(timestamp_ms: int | None) -> str:
    """Unix 毫秒转 ISO 8601（UTC）；为空则用当前时间。"""
    if timestamp_ms is None:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def save_memories(
    request_id: str,
    user_id: str,
    session_id: str,
    messages: list[tuple[str, str, int | None]],
    vectors: list[list[float]] | None,
    metadata_list: list[dict] | None = None,
    kinds: list[str] | None = None,
    source_ids: list[str] | None = None,
) -> list[str]:
    """写入一批记忆（调用方保证 request_id 未写入过、messages 与 vectors 等长）。

    Args:
        messages: (role, content, timestamp_ms) 元组列表，content 已保证非空。
        vectors: 与 messages 一一对应的 embedding 向量。
            为 None 时表示 embedding 服务不可用，仅写 SQLite（BM25 仍可检索），
            不写 Qdrant 向量，保证 /add 不因向量化失败而丢数据或 500。
        metadata_list: 与 messages 一一对应的分类元数据，每项含
            type/entities/time_expr/is_correction。为 None 时使用默认值。
        kinds: 与 messages 一一对应的记忆类别（'chunk' | 'fact'）。
            为 None 时全部按 'chunk' 处理（向后兼容旧调用）。
        source_ids: 与 messages 一一对应的来源分块 memory_id（事实专用）；
            为 None 时分块自身写 NULL。

    Returns:
        写入的 memory_id 列表（顺序与输入一致）。
    """
    if vectors is not None and len(messages) != len(vectors):
        raise ValueError("messages and vectors length mismatch")
    if metadata_list is not None and len(metadata_list) != len(messages):
        raise ValueError("metadata_list and messages length mismatch")
    if kinds is not None and len(kinds) != len(messages):
        raise ValueError("kinds and messages length mismatch")
    if source_ids is not None and len(source_ids) != len(messages):
        raise ValueError("source_ids and messages length mismatch")

    rows: list[tuple] = []
    points: list[PointStruct] = []
    memory_ids: list[str] = []
    point_ids: list[str] = []
    point_vectors: list[list[float]] = []

    for i, (role, content, timestamp_ms) in enumerate(messages):
        vector = vectors[i] if vectors is not None else None
        memory_id = uuid.uuid4().hex
        memory_ids.append(memory_id)
        created_at = _iso_from_ms(timestamp_ms)
        meta = metadata_list[i] if metadata_list else {}
        entities_json = json.dumps(
            [_coerce_str(e) for e in (meta.get("entities") or []) if e is not None],
            ensure_ascii=False,
        )
        # 防御：SQLite 只能绑定 str/int/float/bytes/None；time_expr/type 统一转 str
        time_expr = meta.get("time_expr")
        if time_expr is not None and not isinstance(time_expr, str):
            time_expr = json.dumps(time_expr, ensure_ascii=False) if isinstance(
                time_expr, (dict, list)
            ) else str(time_expr)
        memory_type = meta.get("type")
        if not isinstance(memory_type, str):
            memory_type = "other"
        kind = kinds[i] if kinds else "chunk"
        if not isinstance(kind, str) or not kind:
            kind = "chunk"
        source_id = source_ids[i] if source_ids else None
        rows.append(
            (
                memory_id,
                _coerce_str(request_id),
                _coerce_str(user_id),
                _coerce_str(session_id),
                _coerce_str(role),
                _coerce_str(content),
                created_at,
                memory_type,
                entities_json,
                time_expr,
                1 if meta.get("is_correction") else 0,
                kind,
                source_id,
            )
        )
        if vector is not None:
            point_ids.append(memory_id)
            point_vectors.append(vector)
            points.append(
                PointStruct(
                    id=memory_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "session_id": session_id,
                        "request_id": request_id,
                        "role": role,
                        "created_at": created_at,
                        "kind": kind,
                    },
                )
            )

    if not rows:
        return []

    # 先写 SQLite（幂等事实源），再写 Qdrant（waiting 模式同步落盘）
    with _db_lock, _get_conn() as conn:
        conn.executemany(
            "INSERT INTO memories "
            "(id, request_id, user_id, session_id, role, content, created_at, "
            "type, entities, time_expr, is_correction, kind, source_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()

    if points:
        client = _ensure_qdrant()
        with _q_lock:
            client.upsert(
                collection_name=config.QDRANT_COLLECTION,
                points=points,
                wait=True,  # 同步等待持久化完成，保证写入后立即可检索
            )
        # 同步更新内存向量缓存，保证 /add 后立即能被 /search 召回
        _vc_append(
            point_ids,
            point_vectors,
            [str(_coerce_str(user_id))] * len(point_ids),
        )
    logger.info(
        "stored %d memories for request_id=%s user_id=%s (vectors=%s)",
        len(rows),
        request_id,
        user_id,
        "yes" if points else "skipped",
    )
    return memory_ids


def search_memories(
    user_id: str, query_vector: list[float], top_k: int
) -> list[dict]:
    """按 user_id 隔离的向量检索，结果按相关性降序，条数不超过 top_k。"""
    cached = _vc_search(user_id, query_vector, top_k)
    if cached is not None:
        scored = cached
    else:
        client = _ensure_qdrant()
        with _q_lock:
            hits = client.query_points(
                collection_name=config.QDRANT_COLLECTION,
                query=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=top_k,
                with_payload=False,
            ).points
        scored = [
            (str(hit.id), float(hit.score) if hit.score is not None else None)
            for hit in hits
        ]

    if not scored:
        return []

    meta_map = get_memories_by_ids([mid for mid, _ in scored])

    results: list[dict] = []
    for memory_id, score in scored:
        row = meta_map.get(memory_id)
        # Qdrant 命中但 SQLite 无记录（异常状态）时跳过，保证契约要求非空 content
        if row is None or not row["content"]:
            continue
        results.append(
            {
                "id": memory_id,
                "content": row["content"],
                "score": score,
                "created_at": row["created_at"],
                "type": row.get("type", "other"),
                "entities": row.get("entities", []),
                "time_expr": row.get("time_expr"),
                "is_correction": bool(row.get("is_correction", 0)),
                "superseded": int(row.get("superseded", 0)),
                "importance": float(row.get("importance", 1.0)),
                "kind": row.get("kind", "chunk"),
            }
        )
    return results


# ---------- 链路 2~5 辅助查询 ----------


def _row_to_dict(row: sqlite3.Row) -> dict:
    """把 memories 表行转成 dict，entities 反序列化为 list。"""
    d = dict(row)
    raw_entities = d.get("entities")
    if isinstance(raw_entities, str):
        try:
            d["entities"] = json.loads(raw_entities)
        except json.JSONDecodeError:
            d["entities"] = []
    return d


def get_all_memories() -> list[dict]:
    """读取全部记忆（用于启动时构建 BM25 索引）。"""
    with _db_lock, _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, user_id, content, created_at, type, entities, "
            "time_expr, is_correction, superseded, importance FROM memories"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_memories_by_user(user_id: str) -> list[dict]:
    """读取某用户的全部记忆（用于 BM25 增量重建）。

    v4 P0：排除 kind='summary'。摘要是给 F1 全局题用的抽取式概览，
    不应进入 BM25 索引去参与普通相关性检索（会稀释 top-k 结果）。
    """
    with _db_lock, _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, user_id, content, created_at FROM memories "
            "WHERE user_id = ? AND COALESCE(kind, 'chunk') != 'summary'",
            (user_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def list_sessions() -> list[tuple[str, str]]:
    """列出库内全部 (user_id, session_id) 组合（离线回填摘要用）。"""
    with _db_lock, _get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT user_id, session_id FROM memories "
            "WHERE COALESCE(kind, 'chunk') != 'summary'"
        ).fetchall()
    return [(r["user_id"], r["session_id"]) for r in rows]


def get_memories_by_session(user_id: str, session_id: str) -> list[dict]:
    """读取某用户某 session 的全部未取代记忆（含 kind），用于生成会话摘要。"""
    with _db_lock, _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, content, created_at, kind FROM memories "
            "WHERE user_id = ? AND session_id = ? "
            "AND content IS NOT NULL AND content != '' "
            "AND COALESCE(superseded, 0) = 0 "
            "ORDER BY created_at",
            (user_id, session_id),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_memories_for_summary(user_id: str) -> list[dict]:
    """读取某用户全部未取代记忆（含 kind），用于摘要/全局型检索。

    与 get_memories_by_user 的区别：额外返回 kind，并按内容非空、未取代过滤，
    保证返回条目满足契约（content 非空）。
    """
    with _db_lock, _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, content, created_at, kind FROM memories "
            "WHERE user_id = ? AND content IS NOT NULL AND content != '' "
            "AND COALESCE(superseded, 0) = 0",
            (user_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_memories_by_ids(memory_ids: list[str]) -> dict[str, dict]:
    """按 ID 批量读取记忆完整元数据，返回 {id: dict}。"""
    if not memory_ids:
        return {}
    placeholders = ",".join("?" for _ in memory_ids)
    with _db_lock, _get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        ).fetchall()
    return {r["id"]: _row_to_dict(r) for r in rows}


def mark_superseded(memory_ids: list[str]) -> int:
    """把指定记忆标记为已被取代（superseded=1）。"""
    if not memory_ids:
        return 0
    placeholders = ",".join("?" for _ in memory_ids)
    with _db_lock, _get_conn() as conn:
        cur = conn.execute(
            f"UPDATE memories SET superseded = 1 WHERE id IN ({placeholders})",
            memory_ids,
        )
        conn.commit()
        return cur.rowcount


def bump_importance(memory_ids: list[str], delta: float, cap: float) -> None:
    """惰性 importance 增强：被命中返回的记忆 importance += delta，封顶 cap。

    TODO: 后续可加时间衰减定时任务（如每周对长期未命中记忆 importance * 0.95）。
    """
    if not memory_ids:
        return
    placeholders = ",".join("?" for _ in memory_ids)
    with _db_lock, _get_conn() as conn:
        conn.execute(
            f"UPDATE memories SET importance = MIN(importance + ?, ?) "
            f"WHERE id IN ({placeholders})",
            (delta, cap, *memory_ids),
        )
        conn.commit()


def find_supersede_candidates(
    user_id: str, entities: list[str], exclude_ids: list[str]
) -> list[dict]:
    """查找可能被纠正的旧记忆：同 user_id、未 superseded、entities 有重叠。

    重叠判定在 SQLite 层用 LIKE 粗筛（entities 是 JSON 数组字符串），
    精确重叠由调用方（治理模块）判定。
    """
    if not entities:
        return []
    clauses = " OR ".join("entities LIKE ?" for _ in entities)
    params: list = [f"%{e}%" for e in entities]
    sql = (
        f"SELECT * FROM memories WHERE user_id = ? AND superseded = 0 "
        f"AND ({clauses})"
    )
    if exclude_ids:
        placeholders = ",".join("?" for _ in exclude_ids)
        sql += f" AND id NOT IN ({placeholders})"
        params.extend(exclude_ids)
    with _db_lock, _get_conn() as conn:
        rows = conn.execute(sql, [user_id, *params]).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------- P1：事实去重 ----------


def find_similar_fact(
    user_id: str, query_vector: list[float], threshold: float
) -> dict | None:
    """返回同 user 下与 query_vector 相似度 ≥ threshold 的最近一条事实。

    与旧的 has_similar_fact 判定等价，但额外带回命中事实的正文（content），
    供上层做「数值/时间签名冲突」判定（v2②：冲突变体不合并）。
    仅比对 kind='fact' 的向量（依赖 Qdrant payload 中的 kind 字段；
    旧数据无 kind，视为非事实，不影响）。任何异常由调用方兜底。

    快路径：走内存向量缓存（_vc_search，单次 ~ms）。本地 Qdrant 带 payload
    过滤的查询会全库逐点扫描（每条 1~2s，且被内部锁串行化），是 /add 的
    facts 阶段耗时 100s+ 的根因。缓存未就绪时回落 Qdrant。
    """
    cached = _vc_search(user_id, query_vector, 50)
    if cached is not None:
        cands = [(mid, score) for mid, score in cached if score >= threshold]
        if not cands:
            return None
        rows = get_memories_by_ids([mid for mid, _ in cands])
        for mid, score in cands:
            row = rows.get(mid) or {}
            if row.get("kind") == "fact":
                return {
                    "id": mid,
                    "content": row.get("content") or "",
                    "score": float(score),
                }
        return None

    client = _ensure_qdrant()
    with _q_lock:
        hits = client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="kind", match=MatchValue(value="fact")),
                ]
            ),
            limit=1,
            score_threshold=threshold,
            with_payload=False,
        ).points
    if not hits:
        return None
    memory_id = str(hits[0].id)
    row = get_memories_by_ids([memory_id]).get(memory_id) or {}
    return {
        "id": memory_id,
        "content": row.get("content") or "",
        "score": float(hits[0].score or 0.0),
    }


# ---------- P2：实体关联索引 ----------


def normalize_entity(entity) -> str:
    """实体归一化：转 str、去首尾空白、转小写。空则返回 ""。"""
    if entity is None:
        return ""
    return str(entity).strip().lower()


def add_entity_index_batch(
    user_id: str, entries: list[tuple[str, list[str]]]
) -> None:
    """批量写入实体索引。entries 为 [(memory_id, entities), ...]。

    entity 归一化后写入；重复 (user_id, entity, memory_id) 由唯一索引忽略。
    """
    rows: list[tuple] = []
    for memory_id, entities in entries:
        if not memory_id:
            continue
        for e in entities or []:
            ne = normalize_entity(e)
            if ne:
                rows.append((ne, memory_id, user_id))
    if not rows:
        return
    with _db_lock, _get_conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO entity_index (entity, memory_id, user_id) "
            "VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()


def find_memory_ids_by_entity(
    user_id: str, entity: str, limit: int, exclude_ids: list[str] | None = None
) -> list[str]:
    """按单个实体查询关联 memory_id（created_at 降序，最新优先）。

    exclude_ids 在 Python 层过滤（避免 SQLite 变量数上限）。
    """
    ne = normalize_entity(entity)
    if not ne or limit <= 0:
        return []
    excluded = set(exclude_ids or [])
    # 多取一些以抵消被 exclude 的部分
    fetch_limit = limit + len(excluded)
    with _db_lock, _get_conn() as conn:
        rows = conn.execute(
            "SELECT e.memory_id AS mid FROM entity_index e "
            "JOIN memories m ON m.id = e.memory_id "
            "WHERE e.user_id = ? AND e.entity = ? AND m.superseded = 0 "
            "ORDER BY m.created_at DESC, e.memory_id "
            "LIMIT ?",
            (user_id, ne, fetch_limit),
        ).fetchall()
    ids = [r["mid"] for r in rows if r["mid"] not in excluded]
    return ids[:limit]
