"""BM25 内存索引：基于 rank_bm25 + jieba，按 user_id 隔离。

- 启动时从 SQLite 加载全部记忆建索引
- Add 时增量更新（rebuild 该用户的索引）
- Search 时按用户查询 top_k
"""

import logging
import threading
from collections import defaultdict
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

import config
import store

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# user_id -> {"memory_ids": [...], "tokenized": [...], "bm25": BM25Okapi}
_user_index: dict[str, dict[str, Any]] = defaultdict(dict)

# user_id -> 递增代号，用于并发重建时只让最新一次的快照落地
_user_gen: dict[str, int] = {}


def _tokenize(text: str) -> list[str]:
    """jieba 分词，转小写，过滤空白。"""
    return [t.strip().lower() for t in jieba.cut(text) if t.strip()]


def init_bm25() -> None:
    """从 SQLite 加载全部记忆，按 user_id 构建 BM25 索引。

    分词（jieba）是全量建索引的主要耗时（14k 条记忆约数十秒）。
    这里先在锁外完成各用户的索引构建，最后一次性在锁内合并，
    避免建索引期间 /search、/add 被全局锁阻塞。
    """
    if not config.ENABLE_BM25:
        logger.info("BM25 disabled by config")
        return

    all_memories = store.get_all_memories()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for m in all_memories:
        grouped[m["user_id"]].append(m)

    built: dict[str, dict[str, Any]] = {}
    for user_id, memories in grouped.items():
        entry = _build_entry(memories)
        if entry is not None:
            built[user_id] = entry

    with _lock:
        # 建索引期间若该用户已被 /add 增量重建过，则保留更新的那份
        for user_id, entry in built.items():
            if user_id not in _user_index:
                _user_index[user_id] = entry

    logger.info("BM25 index initialized for %d users", len(_user_index))


def init_bm25_async() -> None:
    """后台线程建索引：先让 HTTP 服务可用（/health 立即可达），
    建成前检索退化为仅向量召回，避免启动期长时间不可用。"""
    if not config.ENABLE_BM25:
        logger.info("BM25 disabled by config")
        return

    def _run() -> None:
        try:
            init_bm25()
        except Exception:  # noqa: BLE001
            logger.exception("BM25 background init failed")

    threading.Thread(target=_run, name="bm25-init", daemon=True).start()


def _build_entry(memories: list[dict]) -> dict[str, Any] | None:
    """由某用户的全部记忆构建一份索引（不持锁，便于后台/增量复用）。"""
    if not memories:
        return None
    memory_ids = [m["id"] for m in memories]
    tokenized = [_tokenize(m["content"]) for m in memories]
    return {
        "memory_ids": memory_ids,
        "tokenized": tokenized,
        "bm25": BM25Okapi(tokenized),
    }


def add_memories(user_id: str, memories: list[dict]) -> None:
    """Add 时增量更新：重建该用户的索引。

    重建（SELECT 该用户全部记忆 + jieba 分词 + BM25Okapi）放到锁外执行，
    锁内只做一次指针替换。原实现在 _lock 内重建，全量分词期间会把
    /search 和其它用户的 /add 一起挡住（summary+bm25 段 max 34.9s 的来源）。
    """
    if not config.ENABLE_BM25:
        return
    with _lock:
        _user_gen[user_id] = _user_gen.get(user_id, 0) + 1
        gen = _user_gen[user_id]

    entry = _build_entry(store.get_memories_by_user(user_id))

    with _lock:
        # 重建期间若有更新的重建完成，以更新的那份为准，避免旧快照覆盖新索引
        if _user_gen.get(user_id, 0) != gen:
            return
        if entry is None:
            _user_index.pop(user_id, None)
        else:
            _user_index[user_id] = entry


def _ensure_user_index(user_id: str) -> dict[str, Any] | None:
    """取某用户索引；缺失时按需从 SQLite 构建。

    服务重启后全量后台建索引需要较长时间，期间老用户的索引为空，
    BM25 这一路（以及 embedding 降级时的兜底检索）会静默失效。
    这里按用户懒加载，只对该用户做一次分词，成本可控。
    """
    with _lock:
        entry = _user_index.get(user_id)
        if entry and "bm25" in entry:
            return entry

    entry = _build_entry(store.get_memories_by_user(user_id))
    if entry is None:
        return None

    with _lock:
        # 构建期间可能已被 /add 或并发请求重建，保留已有那份
        existing = _user_index.get(user_id)
        if existing and "bm25" in existing:
            return existing
        _user_index[user_id] = entry
    return entry


def search(user_id: str, query: str, top_k: int) -> list[tuple[str, float]]:
    """按 user_id 查询 BM25 得分，返回 [(memory_id, score)] 降序列表。"""
    if not config.ENABLE_BM25:
        return []

    entry = _ensure_user_index(user_id)
    if not entry or "bm25" not in entry:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    with _lock:
        scores = entry["bm25"].get_scores(tokenized_query)
        memory_ids = entry["memory_ids"]

    # 取 top_k（不持有锁，只读快照）
    indexed = list(zip(memory_ids, scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:top_k]
