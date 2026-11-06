"""FastAPI 入口：POST /add、POST /search、GET /health。

AML 平台契约要点：
- /add 同步写入，幂等（同 request_id 去重仍返回 200 + 原样 ID）
- /search 返回 {"data": [...]}，条数不超过 top_k，严格按 user_id 隔离
- /health 无鉴权
- 配置了 SERVICE_API_KEY 时 /add 与 /search 校验 X-Api-Key 头

链路 2~5 管线（各能力均可通过 config 开关降级）：
- Add：语义分块(chunker) -> embedding -> LLM 预分类(classifier) -> 存分块(kind='chunk')
        -> 分块 entities 写实体索引(P2) -> 原子事实抽取(fact_extractor, kind='fact')
        -> 事实去重/入库 -> 事实 entities 写实体索引 -> BM25 增量索引 -> 冲突治理
- Search：query 理解(query_analyzer) -> MCQ 选项级向量化（链路 4）
        -> 向量 top-50 + BM25 top-50 混合召回 -> 去重合并
        -> 实体关联多跳扩展(P2) -> 路由过滤 -> 治理过滤 superseded
        -> LLM rerank(RERANK_MODE) -> importance 加权 -> 事实加成(P3) -> top_k
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date, datetime
from logging.handlers import RotatingFileHandler

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import bm25_index
import chunker
import classifier
import config
import embedder
import fact_derive
import fact_extractor
import fusion
import governance
import jieba
import query_analyzer
import reranker
import store
import summarizer
import time_normalizer
from models import (
    AddRequest,
    AddResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- API I/O 日志（/add、/search 入参出参留痕） ----------

# JSON Lines：每行一条记录，direction=request/response，按 request_id 或 query 配对。
_api_io_logger = logging.getLogger("api_io")
_api_io_full_logger = logging.getLogger("api_io_full")


def _init_api_io_logging() -> None:
    """挂载轮转文件处理器；失败时降级为关闭，绝不影响服务启动。"""
    # main.py 会被执行两次（__main__ 与 uvicorn 导入的 main），而 logger 按名字全局共享，
    # 这里做幂等保护，避免同一文件被挂多个 handler 导致每行日志重复写入。
    if _api_io_logger.handlers or _api_io_full_logger.handlers:
        return

    for lg in (_api_io_logger, _api_io_full_logger):
        lg.propagate = False
        lg.setLevel(logging.INFO)

    if not config.API_IO_LOG:
        _api_io_logger.disabled = True
        _api_io_full_logger.disabled = True
        return

    try:
        os.makedirs(config.API_IO_LOG_DIR, exist_ok=True)
        handlers = [
            (
                _api_io_logger,
                "api_io.jsonl",
                config.API_IO_LOG_MAX_BYTES,
                config.API_IO_LOG_BACKUPS,
            ),
            (
                _api_io_full_logger,
                "api_io_full.jsonl",
                config.API_IO_FULL_MAX_BYTES,
                config.API_IO_FULL_BACKUPS,
            ),
        ]
        for lg, filename, max_bytes, backups in handlers:
            handler = RotatingFileHandler(
                os.path.join(config.API_IO_LOG_DIR, filename),
                maxBytes=max_bytes,
                backupCount=backups,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            lg.addHandler(handler)
    except Exception as exc:  # noqa: BLE001
        logger.warning("api io log init failed, disabled: %s", type(exc).__name__)
        _api_io_logger.disabled = True
        _api_io_full_logger.disabled = True
        return

    _api_io_full_logger.disabled = not config.API_IO_LOG_FULL


_init_api_io_logging()


def _client_ip(request: Request) -> str:
    """优先取代理头（DevBridge 隧道下 request.client 恒为 127.0.0.1）。"""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else "unknown"


def _preview(text: str, n: int | None = None) -> str:
    text = text or ""
    limit = n if n is not None else config.API_IO_PREVIEW_CHARS
    return text if len(text) <= limit else f"{text[:limit]}...(+{len(text) - limit})"


def _emit(lg: logging.Logger, payload: dict) -> None:
    try:
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            **payload,
        }
        lg.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001
        # 日志失败绝不影响接口
        pass


def _log_add_input(request: Request, req: AddRequest) -> None:
    if not config.API_IO_LOG:
        return
    _emit(_api_io_logger, {
        "endpoint": "/add",
        "direction": "request",
        "client_ip": _client_ip(request),
        "ua": (request.headers.get("user-agent") or "")[:120],
        "forwarded_host": request.headers.get("x-forwarded-host") or "",
        "request_id": req.request_id,
        "user_id": req.user_id,
        "session_id": req.session_id,
        "n_messages": len(req.messages),
        "total_chars": sum(len(m.content or "") for m in req.messages),
        "first_message": _preview(req.messages[0].content if req.messages else ""),
    })
    if config.API_IO_LOG_FULL:
        _emit(_api_io_full_logger, {
            "endpoint": "/add",
            "direction": "request",
            "request_id": req.request_id,
            "body": req.model_dump(),
        })


def _log_add_output(
    request: Request,
    req: AddRequest,
    elapsed_ms: float,
    status_text: str,
    *,
    duplicate: bool = False,
    n_chunks: int = 0,
    memory_ids: list[str] | None = None,
) -> None:
    if not config.API_IO_LOG:
        return
    ids = memory_ids or []
    _emit(_api_io_logger, {
        "endpoint": "/add",
        "direction": "response",
        "client_ip": _client_ip(request),
        "request_id": req.request_id,
        "user_id": req.user_id,
        "session_id": req.session_id,
        "status": status_text,
        "duplicate": duplicate,
        "n_chunks": n_chunks,
        "n_ids": len(ids),
        "ids": ids[:50],
        "elapsed_ms": round(elapsed_ms, 1),
    })
    if config.API_IO_LOG_FULL:
        _emit(_api_io_full_logger, {
            "endpoint": "/add",
            "direction": "response",
            "request_id": req.request_id,
            "status": status_text,
            "duplicate": duplicate,
            "n_chunks": n_chunks,
            "memory_ids": ids,
        })


def _log_search_input(request: Request, req: SearchRequest) -> None:
    if not config.API_IO_LOG:
        return
    opts = req.options or []
    _emit(_api_io_logger, {
        "endpoint": "/search",
        "direction": "request",
        "client_ip": _client_ip(request),
        "ua": (request.headers.get("user-agent") or "")[:120],
        "forwarded_host": request.headers.get("x-forwarded-host") or "",
        "user_id": req.user_id,
        "top_k": req.top_k,
        "query": req.query,
        "n_options": len(opts),
        "options": [_preview(o, 80) for o in opts[:5]],
    })
    if config.API_IO_LOG_FULL:
        _emit(_api_io_full_logger, {
            "endpoint": "/search",
            "direction": "request",
            "user_id": req.user_id,
            "body": req.model_dump(),
        })


def _log_search_output(
    request: Request,
    req: SearchRequest,
    elapsed_ms: float,
    status_text: str,
    items: list,
) -> None:
    if not config.API_IO_LOG:
        return
    total_chars = sum(len(getattr(it, "content", "") or "") for it in items)
    top1 = items[0] if items else None
    _emit(_api_io_logger, {
        "endpoint": "/search",
        "direction": "response",
        "client_ip": _client_ip(request),
        "user_id": req.user_id,
        "top_k": req.top_k,
        "query": req.query,
        "status": status_text,
        "n_returned": len(items),
        "total_chars": total_chars,
        "ids": [it.id for it in items],
        "scores": [round(float(it.score), 6) if it.score is not None else None for it in items],
        "top1": {
            "id": top1.id,
            "score": top1.score,
            "created_at": top1.created_at,
            "preview": _preview(top1.content),
        } if top1 else None,
        "elapsed_ms": round(elapsed_ms, 1),
    })
    if config.API_IO_LOG_FULL:
        _emit(_api_io_full_logger, {
            "endpoint": "/search",
            "direction": "response",
            "user_id": req.user_id,
            "query": req.query,
            "status": status_text,
            "elapsed_ms": round(elapsed_ms, 1),
            "body": {"data": [it.model_dump() for it in items]},
        })


# 选项词面重叠时过滤的常见虚词（避免 "the/and/her" 之类造成虚假重叠）
_OVERLAP_STOPWORDS = frozenset({
    "the", "and", "her", "his", "him", "she", "for", "with", "that", "this",
    "her", "was", "were", "are", "has", "had", "have", "from", "into", "out",
    "they", "them", "their", "you", "your", "not", "but", "can", "all", "any",
    "one", "two", "about", "over", "down", "when", "who", "how", "why", "what",
    "where", "which", "some", "more", "also", "very", "just", "like", "get",
})


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_store()
    await embedder.init_client()
    # BM25 全量分词建索引耗时较长，放后台线程：先让 /health、/add、/search 可达，
    # 建成前检索退化为仅向量召回（远好于启动期端口不可用）。
    bm25_index.init_bm25_async()
    logger.info("memory service started (chains 1-5)")
    yield
    await embedder.close_client()
    store.close_store()
    logger.info("memory service stopped")


app = FastAPI(title="Agent Memory Service (Chains 1-5)", lifespan=lifespan)


# ---------- 鉴权 ----------


async def verify_api_key(request: Request) -> None:
    """若配置了 SERVICE_API_KEY，则校验 X-Api-Key 头；未配置则放行。"""
    expected = config.SERVICE_API_KEY
    if not expected:
        return
    provided = request.headers.get("X-Api-Key")
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-Api-Key",
        )


# ---------- 异常处理 ----------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 不返回请求正文，仅返回校验错误概要
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "request validation failed", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled error on %s: %s", request.url.path, type(exc).__name__)
    if request.url.path in ("/add", "/search"):
        _emit(_api_io_logger, {
            "endpoint": request.url.path,
            "direction": "response",
            "client_ip": _client_ip(request),
            "status": f"error:unhandled:{type(exc).__name__}",
        })
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "internal server error"},
    )


# ---------- P1/P2 辅助：事实抽取 & 实体扩展 ----------

# 事实 embedding 分批大小（避免单次过大；单批失败只丢该批，不影响其他）
_FACT_EMBED_BATCH = 32


def _cosine(a: list[float], b: list[float]) -> float:
    """纯 Python 余弦相似度（用于批内事实去重，无需新依赖）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


def _cand_strength(c: dict) -> float:
    """候选记忆的召回强度：优先向量分，其次 BM25 归一化分。"""
    s = c.get("score")
    if s is None:
        s = c.get("bm25_score")
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


async def _extract_and_store_facts(
    req: AddRequest,
    chunks: list[chunker.Chunk],
    chunk_ids: list[str],
) -> None:
    """对每个分块抽取原子事实，去重后作为独立记忆单元入库（kind='fact'）。

    失败隔离：任何异常由调用方捕获，不阻塞 /add。
    """
    if not chunks or not chunk_ids:
        return

    # 1) 并发抽取每个分块的事实。
    #    失败的分块延后补抽：网关抖动时 extract_facts 会整块返回空，
    #    静默丢事实直接导致 A1（单点回忆）与 C1（时间）能力崩塌。
    fact_lists: list[list[dict]] = [[] for _ in chunks]
    pending = list(range(len(chunks)))
    for round_idx in range(max(0, config.FACT_RETRY_ROUNDS) + 1):
        if not pending:
            break
        if round_idx:
            await asyncio.sleep(config.FACT_RETRY_DELAY_SECONDS * round_idx)
        results = await asyncio.gather(
            *(
                fact_extractor.extract_facts(chunks[i].content, raise_on_error=True)
                for i in pending
            ),
            return_exceptions=True,
        )
        failed: list[int] = []
        for i, res in zip(pending, results):
            if isinstance(res, asyncio.CancelledError):
                raise res
            if isinstance(res, BaseException):
                failed.append(i)
            else:
                fact_lists[i] = res
        if failed:
            logger.warning(
                "fact extraction failed for %d/%d chunks (round %d/%d)",
                len(failed),
                len(chunks),
                round_idx + 1,
                config.FACT_RETRY_ROUNDS + 1,
            )
        pending = failed

    # 2) 汇总候选事实（保留所属分块 id 与时间戳）
    candidates: list[dict] = []
    for facts, cid, chunk in zip(fact_lists, chunk_ids, chunks):
        for f in facts:
            candidates.append(
                {
                    "fact": time_normalizer.normalize_time(
                        f["fact"], chunk.start_timestamp
                    ),
                    "entities": f.get("entities") or [],
                    "time": f.get("time"),
                    "type": f.get("type") or "fact",
                    "source_id": cid,
                    "ts": chunk.start_timestamp,
                }
            )

    # 2.1) v2①：确定性派生「相对时间 → 绝对日期」事实（纯规则，无 LLM）
    #      LLM 抽取会把 "半个月前" 原样保留，检索时无法落到具体日期；
    #      这里按分块时间戳把相对表达解析成绝对日期，作为独立 fact 入库。
    if config.ENABLE_FACT_DERIVE:
        for cid, chunk in zip(chunk_ids, chunks):
            try:
                derived = fact_derive.derive_time_facts(
                    chunk.content, chunk.start_timestamp
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("fact derive failed: %s", type(exc).__name__)
                continue
            for d in derived:
                candidates.append(
                    {
                        "fact": d["fact"],
                        "entities": d.get("entities") or [],
                        "time": d.get("time"),
                        "type": d.get("type") or "event",
                        "source_id": cid,
                        "ts": chunk.start_timestamp,
                    }
                )

    if not candidates:
        return

    # 3) 分批 embedding + 去重（批内相似 + 已有事实相似）
    #    v2②：相似但数值/日期签名不同 => 冲突变体，两条都保留（不静默丢新值）
    threshold = config.FACT_DEDUP_THRESHOLD
    accepted: list[dict] = []
    accepted_vectors: list[list[float]] = []
    for start in range(0, len(candidates), _FACT_EMBED_BATCH):
        batch = candidates[start : start + _FACT_EMBED_BATCH]
        try:
            vectors = await embedder.embed_texts([c["fact"] for c in batch])
        except Exception as exc:  # noqa: BLE001
            logger.warning("fact embedding batch failed: %s", type(exc).__name__)
            continue

        for cand, vec in zip(batch, vectors):
            if any(
                _cosine(vec, av) >= threshold
                and not fact_derive.signatures_conflict(cand["fact"], acc["fact"])
                for av, acc in zip(accepted_vectors, accepted)
            ):
                continue
            try:
                # 本地 Qdrant 带过滤查询会全库扫描（每条 1~2s），必须挪出事件循环，
                # 否则单次 /add 的去重循环会把整个服务卡住数十秒。
                hit = await asyncio.to_thread(
                    store.find_similar_fact, req.user_id, vec, threshold
                )
                if hit is not None and not fact_derive.signatures_conflict(
                    cand["fact"], hit.get("content") or ""
                ):
                    continue
            except Exception as exc:  # noqa: BLE001
                # 去重查询失败时不阻断写入（降级为仅批内去重）
                logger.warning("fact dedup query failed: %s", type(exc).__name__)
            accepted.append(cand)
            accepted_vectors.append(vec)

    if not accepted:
        return

    # 4) 事实作为独立记忆入库（kind='fact', source_id=分块 id）
    fact_messages = [("fact", c["fact"], c["ts"]) for c in accepted]
    fact_metadata = [
        {
            "type": c["type"],
            "entities": c["entities"],
            "time_expr": c["time"],
            "is_correction": False,
        }
        for c in accepted
    ]
    fact_ids = await asyncio.to_thread(
        store.save_memories,
        request_id=req.request_id,
        user_id=req.user_id,
        session_id=req.session_id,
        messages=fact_messages,
        vectors=accepted_vectors,
        metadata_list=fact_metadata,
        kinds=["fact"] * len(accepted),
        source_ids=[c["source_id"] for c in accepted],
    )

    # 4.1) v2③：纯口水话事实降 importance（不删，只降低排序竞争力）
    if config.ENABLE_FILLER_IMPORTANCE_DOWNWEIGHT:
        try:
            filler_ids = [
                mid
                for mid, c in zip(fact_ids, accepted)
                if fact_derive.is_filler_fact(c["fact"], c["type"])
            ]
            if filler_ids:
                store.bump_importance(
                    filler_ids,
                    config.FILLER_IMPORTANCE_DELTA,
                    config.GOVERNANCE_IMPORTANCE_CAP,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("filler downweight failed: %s", type(exc).__name__)

    # 5) 事实 entities 写入实体索引
    if config.ENABLE_ENTITY_EXPANSION:
        try:
            await asyncio.to_thread(
                store.add_entity_index_batch,
                req.user_id,
                [
                    (mid, c["entities"])
                    for mid, c in zip(fact_ids, accepted)
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("fact entity index write failed: %s", type(exc).__name__)

    logger.info(
        "extracted %d facts, accepted %d (request_id=%s)",
        len(candidates),
        len(accepted),
        req.request_id,
    )


def _expand_candidates_by_entities(user_id: str, candidates: list[dict]) -> list[dict]:
    """实体关联多跳扩展：按候选实体频次取 top-N，沿实体索引补充关联记忆。

    - 无实体 / 索引为空时安全跳过
    - 补充条数受 ENTITY_EXPAND_LIMIT 限制
    - 补充记忆的分数取「种子记忆强度 × 衰减」，使其可参与后续 rerank 竞争
    """
    if not candidates:
        return candidates

    counter: Counter = Counter()
    strength_by_entity: dict[str, float] = {}
    for c in candidates:
        strength = _cand_strength(c)
        for e in c.get("entities") or []:
            ne = store.normalize_entity(e)
            if not ne:
                continue
            counter[ne] += 1
            if strength > strength_by_entity.get(ne, 0.0):
                strength_by_entity[ne] = strength
    if not counter:
        return candidates

    top_entities = [e for e, _ in counter.most_common(config.ENTITY_EXPAND_TOP)]
    existing_ids = {c["id"] for c in candidates}
    seen_new: set[str] = set()
    added = 0

    for ent in top_entities:
        if added >= config.ENTITY_EXPAND_LIMIT:
            break
        try:
            ids = store.find_memory_ids_by_entity(
                user_id,
                ent,
                config.ENTITY_EXPAND_LIMIT,
                exclude_ids=list(existing_ids | seen_new),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("entity index query failed: %s", type(exc).__name__)
            continue
        if not ids:
            continue

        meta_map = store.get_memories_by_ids(ids)
        ent_strength = strength_by_entity.get(ent, 0.0)
        for mid in ids:
            if added >= config.ENTITY_EXPAND_LIMIT:
                break
            if mid in existing_ids or mid in seen_new:
                continue
            meta = meta_map.get(mid)
            if meta is None or not meta.get("content"):
                continue
            candidates.append(
                {
                    "id": mid,
                    "content": meta["content"],
                    "score": (ent_strength * config.ENTITY_EXPAND_SCORE_DECAY)
                    if ent_strength > 0
                    else None,
                    "bm25_score": None,
                    "created_at": meta.get("created_at"),
                    "type": meta.get("type", "other"),
                    "entities": meta.get("entities", []),
                    "time_expr": meta.get("time_expr"),
                    "is_correction": bool(meta.get("is_correction", 0)),
                    "superseded": int(meta.get("superseded", 0)),
                    "importance": float(meta.get("importance", 1.0)),
                    "kind": meta.get("kind", "chunk"),
                    "expansion": True,
                }
            )
            seen_new.add(mid)
            added += 1

    if added:
        logger.info("entity expansion added %d candidates", added)
    return candidates


def _collect_source_maps(
    hits: list[dict],
) -> tuple[dict[str, dict], dict[str, dict], list[tuple[dict, str]]]:
    """批量取命中元数据 + 命中事实的源分块，供反馈过滤与原文附加共用。"""
    hit_ids = [h["id"] for h in hits if h.get("id")]
    meta_map = store.get_memories_by_ids(hit_ids)

    pending: list[tuple[dict, str]] = []
    for h in hits:
        meta = meta_map.get(h.get("id")) or {}
        kind = h.get("kind") or meta.get("kind")
        source_id = h.get("source_id") or meta.get("source_id")
        if kind == "fact" and source_id:
            pending.append((h, source_id))

    source_map = (
        store.get_memories_by_ids(list({sid for _, sid in pending}))
        if pending
        else {}
    )
    return meta_map, source_map, pending


_FEEDBACK_MARK = "Streaming online QA feedback"
_FEEDBACK_SCORE_RE = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)')


def _feedback_score(content: str) -> float | None:
    """解析在线 QA 反馈记忆的 score；非反馈记忆返回 None。"""
    if _FEEDBACK_MARK not in (content or ""):
        return None
    m = _FEEDBACK_SCORE_RE.search(content)
    return float(m.group(1)) if m else None


def _apply_bad_feedback_filter(
    hits: list[dict],
    meta_map: dict[str, dict],
    source_map: dict[str, dict],
    threshold: float,
) -> list[dict]:
    """过滤被判定为错误的在线 QA 反馈（score < threshold）及其派生事实。

    错误反馈的正文与派生事实（如 "Option E states that ..."）会以高分记忆
    排在首位，诱导 Answer 重复同一错误答案。正例反馈保留，可继续作为经验。
    """
    bad_ids: set[str] = set()
    for h in hits:
        sc = _feedback_score(h.get("content") or "")
        if sc is not None and sc < threshold:
            bad_ids.add(h.get("id"))
    for sid, meta in source_map.items():
        sc = _feedback_score(meta.get("content") or "")
        if sc is not None and sc < threshold:
            bad_ids.add(sid)
    if not bad_ids:
        return hits

    kept = []
    for h in hits:
        if h.get("id") in bad_ids:
            continue
        src = h.get("source_id") or (meta_map.get(h.get("id")) or {}).get("source_id")
        if src in bad_ids:
            continue
        kept.append(h)
    return kept


def _integrate_source_passages(
    hits: list[dict],
    source_ctx: tuple[dict[str, dict], dict[str, dict], list[tuple[dict, str]]] | None = None,
) -> None:
    """[Passage Integration] 把命中事实的源分块原文附加到其 content（就地修改）。

    HippoRAG 2 思路：事实在抽取时可能丢失日期等细节，返回时把 source_id
    指向的原始分块一并带上，既有精准事实、又有完整上下文。

    为避免上下文膨胀（同一分块被多条事实重复附加，单条事实最多撑到 2k 字符），
    提供三项约束：
    - 源分块本身已在返回结果中时跳过（原文已经出现，不必重复）
    - 同一源分块只给排位最高的一条事实附加（hits 已按分数降序）
    - 全局附件总预算 PASSAGE_TOTAL_BUDGET

    - 只处理 kind='fact' 且能解析出 source_id 的记忆；其余原样不变
    - 两次批量查询（命中元数据 + 源分块正文），避免 N+1
    - 源分块查不到时保持原 fact content 不变
    """
    if not hits:
        return

    # 1) 按命中 id 批量取元数据，补齐向量命中路径缺失的 kind/source_id
    #    （调用方已取过时复用，避免重复查询）
    if source_ctx is not None:
        meta_map, source_map, pending = source_ctx
    else:
        meta_map, source_map, pending = _collect_source_maps(hits)
    if not pending:
        return

    # 3) 拼接：{fact内容}\n[原文] {源分块内容（截断）}
    max_chars = config.PASSAGE_MAX_CHARS
    budget = config.PASSAGE_TOTAL_BUDGET
    hit_id_set = {h.get("id") for h in hits}
    used_chars = 0
    done_sources: set[str] = set()
    for h, source_id in pending:
        if config.PASSAGE_SKIP_IF_IN_HITS and source_id in hit_id_set:
            continue
        if config.PASSAGE_ONCE_PER_SOURCE and source_id in done_sources:
            continue
        src = source_map.get(source_id)
        passage = src.get("content") if src else None
        if not passage:
            continue
        if max_chars and max_chars > 0 and len(passage) > max_chars:
            passage = passage[:max_chars]
        if budget > 0 and used_chars + len(passage) > budget:
            # 预算耗尽：后续事实不再附加原文（返回的仍是干净事实）
            break
        h["content"] = f"{h['content']}\n[原文] {passage}"
        used_chars += len(passage)
        done_sources.add(source_id)


def _normalize_scores(items: list[dict]) -> None:
    """把 final_score 归一到 [0,1]（仅当最大值 >1 时等比缩放，保持排序不变）。"""
    if not items:
        return
    top = max((h.get("final_score") or 0.0) for h in items)
    if top <= 1.0 or top <= 0:
        return
    for h in items:
        h["final_score"] = float(h.get("final_score") or 0.0) / top


def _trim_by_budget(hits: list[dict], budget: int, min_items: int) -> list[dict]:
    """按分数序累计字符数，超出预算即截断（至少保留 min_items 条）。"""
    if budget <= 0 or not hits:
        return hits
    kept: list[dict] = []
    used = 0
    for h in hits:
        size = len(h.get("content") or "")
        if kept and len(kept) >= min_items and used + size > budget:
            break
        kept.append(h)
        used += size
    return kept


_SUMMARY_QUERY_RE = re.compile(
    r"(?:\b(summary|summarize|summarise|overview|recap)\b[\s\S]{0,40}?"
    r"\b(everything|all|entire|whole)\b)"
    r"|(总结|汇总|概述|回顾|整体情况)[\s\S]{0,10}?(全部|所有|整个|整体|一切)",
    re.IGNORECASE,
)


def _is_summary_request(req: SearchRequest) -> bool:
    """判断是否为摘要/全局综合型请求（需要全量覆盖而非 top-k 相关性）。

    v4 P0：官方 harness 固定 top_k=100，旧的「大 top_k + 关键词」双门控
    在正式评测中永不命中（SUMMARY_TOP_K_THRESHOLD=300）。打开
    SUMMARY_GATE_BY_INTENT 后改为纯意图触发；是否真的走摘要返回由
    _build_summary_items 决定（无摘要记忆层时退化为普通检索）。
    """
    if not config.ENABLE_SUMMARY_MODE:
        return False
    keyword_hit = bool(_SUMMARY_QUERY_RE.search(req.query or ""))
    if config.SUMMARY_GATE_BY_INTENT:
        return keyword_hit
    big_k = req.top_k >= config.SUMMARY_TOP_K_THRESHOLD
    if config.SUMMARY_REQUIRE_KEYWORD:
        return big_k and keyword_hit
    return big_k or keyword_hit


def _build_summary_items(req: SearchRequest) -> list[SearchResultItem]:
    """全局模式：返回该用户全部未取代记忆。

    排序策略：分块（完整上下文）优先 + 各自按 created_at 升序（时间线），
    再追加原子事实，保证摘要模型能覆盖全历史；受 top_k 与字符预算约束。
    """
    rows = store.get_memories_for_summary(req.user_id)
    if not rows:
        return []

    # v4 P0：库里还没有摘要记忆层（kind='summary'）时，退化到普通检索。
    # 否则会把上百条 chunk 硬塞进 top_k，把相关性排序结果挤掉。
    if not any(r.get("kind") == "summary" for r in rows):
        return []

    def _ts(r: dict) -> str:
        return r.get("created_at") or ""

    # v4 P0：会话摘要优先（高密度概览，直接覆盖 F1 的时间线与关键事实），
    # 再追加原始分块（完整上下文）与原子事实，均按时间线升序。
    summaries = sorted((r for r in rows if r.get("kind") == "summary"), key=_ts)
    chunks = sorted((r for r in rows if r.get("kind") == "chunk"), key=_ts)
    facts = sorted((r for r in rows if r.get("kind") == "fact"), key=_ts)

    limit = max(1, int(req.top_k))
    budget = config.SUMMARY_CHARS_BUDGET
    items: list[SearchResultItem] = []
    used = 0
    for r in summaries + chunks + facts:
        if len(items) >= limit:
            break
        content = r.get("content") or ""
        if not content:
            continue
        if budget > 0 and items and used + len(content) > budget:
            continue
        items.append(
            SearchResultItem(
                id=r["id"],
                content=content,
                score=round(max(0.01, 1.0 - len(items) * 0.001), 6),
                created_at=r.get("created_at"),
            )
        )
        used += len(content)
    return items



def _apply_recall_floor(req: SearchRequest, hits: list[dict]) -> list[dict]:
    """v5 P0-c 召回保底：常规检索几乎无返回时，用摘要 + 历史分块补齐。

    官方 F1 的 query 不含摘要措辞，意图门控不会命中，因此这里不看 query 措辞，
    只看「这次检索本身是否失败」：返回条数 <= RECALL_FLOOR_MAX_HITS 或最高分
    低于 RECALL_FLOOR_MIN_TOP_SCORE 时，才用摘要层补齐。正常题目（返回条数多
    且分数正常）完全不受影响。

    补齐时对摘要池做等距抽样，保证覆盖整段历史而不是只取最早的几个 session。
    """
    if not config.ENABLE_RECALL_FLOOR:
        return hits
    if len(hits) > config.RECALL_FLOOR_MAX_HITS:
        return hits
    if hits:
        top_score = max(
            float(
                h.get("final_score")
                if h.get("final_score") is not None
                else (h.get("score") or 0.0)
            )
            for h in hits
        )
        if top_score >= config.RECALL_FLOOR_MIN_TOP_SCORE:
            return hits

    # 前提：该用户历史本身要够厚。返回少但库里也就几条，属于「答案不存在」，
    # 补摘要只会把噪声塞进 Answer 上下文。
    try:
        if len(store.get_memories_by_user(req.user_id)) < config.RECALL_FLOOR_MIN_STORE_ROWS:
            return hits
    except Exception as exc:  # noqa: BLE001
        logger.warning("recall floor size check failed: %s", type(exc).__name__)
        return hits

    try:
        pool = _build_summary_items(req)
    except Exception as exc:  # noqa: BLE001
        logger.warning("recall floor failed: %s", type(exc).__name__)
        return hits
    if not pool:
        return hits

    target = min(max(1, int(req.top_k)), max(config.RECALL_FLOOR_FILL, len(hits)))
    if len(pool) > target:
        step = len(pool) / float(target)
        pool = [pool[int(i * step)] for i in range(target)]

    kept_ids = {h.get("id") for h in hits}
    merged = list(hits)
    for it in pool:
        if len(merged) >= target:
            break
        if it.id in kept_ids:
            continue
        kept_ids.add(it.id)
        merged.append(
            {
                "id": it.id,
                "content": it.content,
                "score": it.score,
                "final_score": it.score,
                "created_at": it.created_at,
                "kind": "summary",
            }
        )
    if len(merged) > len(hits):
        logger.info("recall floor: %d -> %d hits", len(hits), len(merged))
    return merged



def _overlap_tokens(text: str) -> set[str]:
    """选项词面重叠用分词：小写、保留长度 >=3 的字母数字 token（过滤常见虚词）。"""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if len(t) >= 3 and t not in _OVERLAP_STOPWORDS}


def _option_overlap(cand_tokens: set[str], option_token_sets: list[set[str]]) -> float:
    """候选内容与各选项的最大词面覆盖率（option token 被覆盖比例，0~1）。"""
    if not cand_tokens or not option_token_sets:
        return 0.0
    best = 0.0
    for os_ in option_token_sets:
        if not os_:
            continue
        ratio = len(cand_tokens & os_) / len(os_)
        if ratio > best:
            best = ratio
    return best


_MONTH_NUM = {
    name: idx
    for idx, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}
_MONTH_ALT = "|".join(sorted(_MONTH_NUM, key=len, reverse=True))

# 查询里的显式日期："September 5, 2025" / "Sep 05 2025" / "2024-01-12"
_QUERY_DATE_RE = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})(?!\d)"
    r"|(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)",
    re.IGNORECASE,
)
# 记忆正文里的事件时间："[Time: Sep 04, 2025, 09:05:32]" / "[2024-01-02 January 2024]"
_CONTENT_DATE_RE = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})(?!\d)"
    r"|(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)",
    re.IGNORECASE,
)


def _parse_query_date(query: str) -> date | None:
    """从 query 中解析唯一的显式日期（需含日+年；只有年月时不处理）。"""
    m = _QUERY_DATE_RE.search(query or "")
    if not m:
        return None
    try:
        if m.group(1):
            return date(int(m.group(3)), _MONTH_NUM[m.group(1).lower()], int(m.group(2)))
        return date(int(m.group(4)), int(m.group(5)), int(m.group(6)))
    except (ValueError, KeyError):
        return None


def _content_dates(content: str) -> list[date]:
    """抽取记忆正文中出现的事件日期（去重；解析失败的跳过）。"""
    found: set[date] = set()
    for m in _CONTENT_DATE_RE.finditer(content or ""):
        try:
            if m.group(1):
                found.add(
                    date(int(m.group(3)), _MONTH_NUM[m.group(1).lower()], int(m.group(2)))
                )
            else:
                found.add(date(int(m.group(4)), int(m.group(5)), int(m.group(6))))
        except (ValueError, KeyError):
            continue
    return list(found)


def _apply_asof_boost(items: list[dict], query: str) -> int:
    """'as of <date>' 查询的状态语义加成：该日期及之前最近事件的记忆优先。

    仅当 query 含「日+年」级显式日期时启用；对每条记忆取其正文里
    <= as_of 的最大日期，越接近 as_of 加成越大（按 ASOF_DECAY_DAYS 衰减）。
    正文没有可用日期的记忆不加成也不惩罚。返回被加成的条数。
    """
    as_of = _parse_query_date(query)
    if as_of is None:
        return 0
    boosted = 0
    for h in items:
        dates = [d for d in _content_dates(h.get("content") or "") if d <= as_of]
        if not dates:
            continue
        delta = (as_of - max(dates)).days
        base = h.get("final_score")
        if base is None:
            base = h.get("score") or 0.0
        h["final_score"] = float(base) + config.ASOF_BOOST / (
            1.0 + delta / max(config.ASOF_DECAY_DAYS, 1.0)
        )
        boosted += 1
    if boosted:
        items.sort(key=lambda x: x.get("final_score") or 0.0, reverse=True)
    return boosted


# 当前状态线索：问"现在 / 目前 / 最近一次"时，新值应覆盖旧值（D1/D2）
_CURRENT_STATE_RE = re.compile(
    r"(现在|目前|当前|如今|最新|最近一次|最近|上一次|上次|后来|最终|最后|"
    r"still|currently|right now|latest|most recent|last time)",
    re.IGNORECASE,
)


def _created_ms(value) -> float | None:
    """ISO8601 created_at -> 毫秒时间戳；解析失败返回 None。"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).timestamp() * 1000.0
    except (TypeError, ValueError):
        return None


def _apply_recency_boost(items: list[dict], query: str) -> int:
    """当前状态类提问的时序加成：created_at 越新加成越大（只加不罚）。

    同一属性存在多条历史陈述时，检索侧对时间不敏感，旧值会压住新值
    （D1 新值覆盖 / D2 矛盾消解失分）。query 含"现在/目前/最近一次"等线索时，
    按 created_at 在候选池内 min-max 归一化：最旧 +0，最新 +RECENCY_BOOST。
    缺失 created_at 的记忆不加成也不惩罚。返回被加成的条数。
    """
    if not items or not _CURRENT_STATE_RE.search(query or ""):
        return 0
    stamps = [_created_ms(h.get("created_at")) for h in items]
    valid = [t for t in stamps if t is not None]
    if len(valid) < 2:
        return 0
    lo, hi = min(valid), max(valid)
    span = hi - lo
    if span <= 0:
        return 0
    boosted = 0
    for h, t in zip(items, stamps):
        if t is None:
            continue
        base = h.get("final_score")
        if base is None:
            base = h.get("score") or 0.0
        h["final_score"] = float(base) + config.RECENCY_BOOST * ((t - lo) / span)
        boosted += 1
    if boosted:
        items.sort(key=lambda x: x.get("final_score") or 0.0, reverse=True)
    return boosted


# 无年份的中文日期：「1月20日」「2月2号」
_CN_MONTH_DAY_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
# 「前后 / 左右」= 近似锚定；「那次 / 当天」= 精确锚定
_DATE_APPROX_RE = re.compile(r"(前后|左右|那几天|around|about|near)", re.IGNORECASE)
_DATE_REF_YEAR = 2000  # 闰年参考年，允许 2/29


def _parse_month_day(query: str) -> tuple[int, int, int] | None:
    """解析无年份的「X月Y日」，返回 (月, 日, 容差天数)。

    仅用于 query 没有年级显式日期时（带年份的走 as-of 逻辑）：
    中文会话里最常见的「1月20日前后」「2月2日那次」在 v2 里解析失败，
    导致 C1 这类"日期定位"问题只能靠字面相似。
    """
    if _parse_query_date(query) is not None:
        return None
    m = _CN_MONTH_DAY_RE.search(query or "")
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    tail = (query or "")[m.end() : m.end() + 12]
    tol = config.DATE_ANCHOR_TOL_DAYS if _DATE_APPROX_RE.search(tail) else 0
    return month, day, tol


def _candidate_dates(h: dict) -> list[date]:
    """候选的可用日期：正文里的事件日期 + 写入时间（session 日期）。"""
    dates = _content_dates(h.get("content") or "")
    ca = h.get("created_at")
    if ca:
        try:
            dates.append(
                datetime.fromisoformat(str(ca).replace("Z", "+00:00")).date()
            )
        except (TypeError, ValueError):
            pass
    return dates


def _month_day_gap(a: date, b: date) -> int:
    """只比月日的环形距离（跨年取较小方向），0~183。"""
    try:
        da = date(_DATE_REF_YEAR, a.month, a.day)
        db = date(_DATE_REF_YEAR, b.month, b.day)
    except ValueError:
        return 999
    gap = abs((da - db).days)
    return min(gap, 366 - gap)


def _apply_date_anchor_boost(items: list[dict], query: str) -> int:
    """「X月Y日（前后）」类查询的日期锚定加成。

    用候选自身的 created_at / 正文日期按 (月, 日) 比对，落在容差内则加成，
    越接近锚定日加成越大；只加不罚。返回被加成的条数。
    """
    if not items:
        return 0
    parsed = _parse_month_day(query)
    if parsed is None:
        return 0
    month, day, tol = parsed
    anchor = date(_DATE_REF_YEAR, month, day)
    boosted = 0
    for h in items:
        gaps = [_month_day_gap(anchor, d) for d in _candidate_dates(h)]
        gaps = [g for g in gaps if g <= tol]
        if not gaps:
            continue
        best = min(gaps)
        strength = 1.0 if tol <= 0 else 1.0 - best / (tol + 1.0)
        base = h.get("final_score")
        if base is None:
            base = h.get("score") or 0.0
        h["final_score"] = float(base) + config.DATE_ANCHOR_BOOST * max(strength, 0.3)
        boosted += 1
    if boosted:
        items.sort(key=lambda x: x.get("final_score") or 0.0, reverse=True)
    return boosted


# 时间类提问线索：需要"时间/频率/次序"作答的问法
_TEMPORAL_QUERY_RE = re.compile(
    r"(什么时候|什么时间|何时|哪一天|哪天|几号|多久|多长时间|隔多久|间隔|"
    r"第一次|最近一次|上一次|上次|"
    r"\bwhen\b|\bwhat date\b|\bwhat day\b|\bhow long\b|\bhow often\b|\bhow many days\b)",
    re.IGNORECASE,
)
# 记忆正文开头的时间戳头（"[2024-01-17 January 2024] "），比对关键词前先剥掉
_LEADING_HEADER_RE = re.compile(r"^\s*\[[^\]]{0,48}\]\s*")
# 附加的源分块原文（"[原文] ..."），参与关键词统计会虚高覆盖
_PASSAGE_TAIL_RE = re.compile(r"\n\s*\[原文\]")


def _query_content_terms(query: str) -> list[str]:
    """query 的关键词：jieba 分词后保留 >=2 字或含数字的实词（去重保序）。"""
    terms: list[str] = []
    for raw in jieba.cut(query or ""):
        t = raw.strip().lower()
        if not t or re.fullmatch(r"[^\w\u4e00-\u9fff]+", t):
            continue
        if len(t) < 2 and not any(ch.isdigit() for ch in t):
            continue
        if t not in terms:
            terms.append(t)
    return terms


def _apply_temporal_coverage_boost(items: list[dict], query: str) -> int:
    """时间类提问的 query 关键词覆盖率加成。

    仅在 query 含"什么时候/多久/间隔"等线索时生效：按关键词在候选正文中的
    覆盖率加权（词权重取候选池内 IDF 近似），把带具体时间陈述的答案顶上来，
    压住句式相近但无时间信息的通用画像事实。只加不罚，未命中关键词的候选
    分数不变。返回被加成的条数。
    """
    if not items or not _TEMPORAL_QUERY_RE.search(query or ""):
        return 0
    terms = _query_content_terms(query)
    if not terms:
        return 0
    bodies = [
        _LEADING_HEADER_RE.sub("", _PASSAGE_TAIL_RE.split(h.get("content") or "")[0]).lower()
        for h in items
    ]
    weights = {}
    for t in terms:
        df = sum(1 for b in bodies if t in b)
        # 候选池内一条都没出现的词（如问句里的"患者"对应库里的"User"）不参与加权
        weights[t] = 1.0 / (1 + df) if df else 0.0
    total = sum(weights.values())
    if total <= 0:
        return 0
    boosted = 0
    for h, body in zip(items, bodies):
        coverage = sum(weights[t] for t in terms if t in body) / total
        if coverage <= 0:
            continue
        base = h.get("final_score")
        if base is None:
            base = h.get("score") or 0.0
        h["final_score"] = float(base) + config.TEMPORAL_COVERAGE_BOOST * coverage
        boosted += 1
    if boosted:
        items.sort(key=lambda x: x.get("final_score") or 0.0, reverse=True)
    return boosted


def _select_diverse(
    items: list[dict], top_k: int, per_source_cap: int
) -> list[dict]:
    """从已排序候选中按多样性挑选 top_k 条。

    以 created_at 视为同一 session：同一 session 最多选 per_source_cap 条，
    超出的近似分块让位给其它 session 的候选，提升跨 session 的证据覆盖
    （对 recall/coverage 型指标有利，避免重复内容挤占名额）。
    per_source_cap<=0 时退化为直接截断（原行为）。
    """
    if top_k <= 0:
        return []
    if per_source_cap <= 0 or len(items) <= top_k:
        return items[:top_k]

    selected: list[dict] = []
    counts: dict[str, int] = {}
    for it in items:
        key = it.get("created_at") or ""
        if counts.get(key, 0) >= per_source_cap:
            continue
        selected.append(it)
        counts[key] = counts.get(key, 0) + 1
        if len(selected) >= top_k:
            return selected

    # 未填满时用被跳过的候选按原顺序补齐（保证返回条数达到 top_k）
    chosen = {id(x) for x in selected}
    for it in items:
        if id(it) in chosen:
            continue
        selected.append(it)
        if len(selected) >= top_k:
            break
    return selected


# 全局综合题线索：问的是「整段历史的整体情况」，需要覆盖而非精确命中
_GLOBAL_QUERY_RE = re.compile(
    r"(summar\w*|recap|overview|taken together|in short|"
    r"all (?:of )?(?:our|my|the) (?:conversations?|sessions?|history|memories)|"
    r"over the course of|throughout (?:our|the)|"
    r"总结|汇总|概述|回顾|总的来说|整体情况|历次|一直以来|全部对话|所有对话)",
    re.IGNORECASE,
)

# 「变化轨迹」线索：需要同时看到旧值与新值，不能把旧值降权
_TRAJECTORY_QUERY_RE = re.compile(
    r"(变化|从.{0,40}?(?:到|变成|成为|增至|降到)|最初|一开始|以前|过去|曾经|"
    r"used to|change[ds]?|no longer|from .{0,40}? to )",
    re.IGNORECASE,
)

# 超过该长度的 query 视为「自带语料的场景/文档题」，其中的 summary/from-to
# 等词只是语料内容，不代表用户在问历史全局或变化轨迹。
_SHORT_QUERY_MAX_CHARS = 600


def _is_short_query(query: str) -> bool:
    return len(query or "") < _SHORT_QUERY_MAX_CHARS


def _is_global_query(query: str) -> bool:
    """全局综合题：需要时间轴覆盖而非 top-k 相关性。"""
    if not config.ENABLE_GLOBAL_COVERAGE:
        return False
    if not _is_short_query(query):
        return False
    return bool(_GLOBAL_QUERY_RE.search(query or ""))


def _is_trajectory_query(query: str) -> bool:
    """「变化轨迹」题：需要旧值与新值同时可见。"""
    if not config.ENABLE_TRAJECTORY_SKIP_CONFLICT:
        return False
    if not _is_short_query(query):
        return False
    return bool(_TRAJECTORY_QUERY_RE.search(query or ""))


_CLAIM_HINT_RE = re.compile(r"\bclaim\s*[:：]", re.IGNORECASE)
_CLAIM_BOILERPLATE_HINTS = (
    "verify the following claim",
    "return one label",
    "not enough info",
    "do not use outside knowledge",
)


def _strip_query_boilerplate(query: str) -> str:
    """去掉 claim 验证类 query 的模板指令头，只保留 Claim 正文用于召回与精排。

    评测 claim 题形如：
      "Verify the following claim using only the supplied corpus. Return one label:
       SUPPORTS, REFUTES, or NOT ENOUGH INFO. ... Do not use outside knowledge.

       Claim: <真正要验证的句子>"
    整段送进 embedding / BM25 时，模板里的 corpus / evidence / sentence / claim 等
    高频词会压过 Claim 正文，把所有 claim 拉到同一条 "Corpus: sentence_xxx" 模板块上
    （实测 202 次 claim 请求中 136 次 top1 是同一条无关记忆，83% 的 top1 分 < 0.5）。
    仅当 Claim 之前确实是指令模板时才截取正文；普通 query 原样返回。
    """
    if not query:
        return query
    m = _CLAIM_HINT_RE.search(query)
    if not m:
        return query
    prefix = query[: m.start()].lower()
    if not any(h in prefix for h in _CLAIM_BOILERPLATE_HINTS):
        return query
    body = query[m.end() :].strip()
    # 正文过短说明是误截（原句里本来就有 "claim:"），保守回退原文
    if len(body) < 20:
        return query
    return body


def _select_temporal_coverage(items: list[dict], top_k: int) -> list[dict]:
    """按时间桶轮转选取：保证返回结果在时间轴上铺开（全局覆盖型问题）。

    先按 created_at 的日期分桶，桶内保持分数序，再逐轮从每个桶取一条，
    使 100 个名额尽量均摊到整段历史，而不是集中在一两个 session。
    """
    if top_k <= 0 or not items:
        return []
    if len(items) <= top_k:
        return items

    buckets: dict[str, list[dict]] = {}
    order: list[str] = []
    for it in items:
        key = str(it.get("created_at") or "")[:10]
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(it)
    order.sort()

    selected: list[dict] = []
    idx = 0
    while len(selected) < top_k:
        progressed = False
        for key in order:
            bucket = buckets[key]
            if idx < len(bucket):
                selected.append(bucket[idx])
                progressed = True
                if len(selected) >= top_k:
                    break
        if not progressed:
            break
        idx += 1
    return selected


# ---------- 接口 ----------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/add",
    response_model=AddResponse,
    dependencies=[Depends(verify_api_key)],
)
async def add_memory(req: AddRequest, request: Request) -> AddResponse:
    t0 = time.perf_counter()
    _log_add_input(request, req)
    n_chunks = 0
    memory_ids: list[str] = []
    degraded = False
    # 分段耗时探针：用于定位 /add 的真实瓶颈（embed / 存储 / 事实抽取 / BM25）
    t_embed = t_store = t_facts = t_bm25 = t0

    # 幂等：同一 request_id 重复写入直接返回成功
    if store.request_exists(req.request_id):
        logger.info("duplicate request_id=%s, skip write", req.request_id)
        _log_add_output(
            request, req, (time.perf_counter() - t0) * 1000, "ok", duplicate=True
        )
        return AddResponse(
            success=True,
            request_id=req.request_id,
            user_id=req.user_id,
            session_id=req.session_id,
        )

    # 1) 过滤空消息
    messages: list[tuple[str, str, int | None]] = [
        (m.role, m.content, m.timestamp)
        for m in req.messages[: config.MAX_MESSAGES_PER_ADD]
        if m.content and m.content.strip()
    ]

    if messages:
        # 2) 语义分块（链路 2 Add 侧；ENABLE_CHUNKING=false 时退化为逐条）
        chunks = chunker.chunk_messages(messages)
        n_chunks = len(chunks)
        chunk_messages_tuples: list[tuple[str, str, int | None]] = [
            (c.role, c.content, c.start_timestamp) for c in chunks
        ]

        # 3) Embedding。失败时不再 500：降级为「仅写 SQLite」（BM25 仍可检索），
        #    避免官方并发 /add 在 embed 排队超时后整条请求被丢弃。
        vectors: list[list[float]] | None
        try:
            vectors = await embedder.embed_texts([m[1] for m in chunk_messages_tuples])
        except Exception as exc:  # noqa: BLE001
            vectors = None
            degraded = True
            logger.warning(
                "embedding failed for request_id=%s: %s; degrade to lexical-only write",
                req.request_id,
                type(exc).__name__,
            )

        t_embed = time.perf_counter()

        # 4) LLM 预分类（链路 3 Add 侧；失败兜底默认值，不阻塞写入）
        metadata_list = await asyncio.gather(
            *(classifier.classify_chunk(m[1]) for m in chunk_messages_tuples)
        )

        # 5) 存储（SQLite + Qdrant）
        try:
            memory_ids = await asyncio.to_thread(
                store.save_memories,
                request_id=req.request_id,
                user_id=req.user_id,
                session_id=req.session_id,
                messages=chunk_messages_tuples,
                vectors=vectors,
                metadata_list=list(metadata_list),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "store failed for request_id=%s: %s",
                req.request_id,
                type(exc).__name__,
            )
            _log_add_output(
                request,
                req,
                (time.perf_counter() - t0) * 1000,
                f"error:store:{type(exc).__name__}",
                n_chunks=n_chunks,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="memory store unavailable",
            ) from exc

        t_store = time.perf_counter()

        # 5.5) [P2] 分块 entities 写入实体索引（失败不阻塞写入）
        if config.ENABLE_ENTITY_EXPANSION:
            try:
                await asyncio.to_thread(
                    store.add_entity_index_batch,
                    req.user_id,
                    [
                        (mid, (meta.get("entities") or []))
                        for mid, meta in zip(memory_ids, metadata_list)
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "chunk entity index write failed: %s", type(exc).__name__
                )

        # 5.6) [P1] 原子事实抽取 -> 去重 -> 入库（kind='fact'，失败不阻塞写入）
        if config.ENABLE_FACT_EXTRACTION:
            try:
                await _extract_and_store_facts(req, chunks, memory_ids)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "fact extraction pipeline failed: %s", type(exc).__name__
                )

        t_facts = time.perf_counter()

        # 5.7) [v4 P0] 摘要记忆层：为本 session 生成/更新一条 kind='summary' 概览。
        #      只写 SQLite、不写向量，不影响普通检索；失败不阻塞写入。
        if config.ENABLE_SUMMARY_LAYER:
            try:
                await asyncio.to_thread(
                    summarizer.rebuild_session_summary, req.user_id, req.session_id
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "summary layer failed: %s", type(exc).__name__
                )

        # 6) BM25 索引增量更新（含事实；失败不阻塞写入）
        try:
            # 重建要对该用户全部记忆做 jieba 分词，同步执行会占住事件循环，
            # 必须放线程池（与 store 的同步调用同理）。
            await asyncio.to_thread(bm25_index.add_memories, req.user_id, [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("bm25 incremental update failed: %s", type(exc).__name__)

        t_bm25 = time.perf_counter()

        # 7) 治理：纠正型记忆 -> 标记被纠正的旧记忆 superseded（失败不阻塞）
        for chunk, meta, mid in zip(chunks, metadata_list, memory_ids):
            if meta.get("is_correction"):
                try:
                    await governance.resolve_correction(
                        user_id=req.user_id,
                        new_content=chunk.content,
                        new_entities=meta.get("entities") or [],
                        new_memory_id=mid,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "governance resolve_correction failed: %s", type(exc).__name__
                    )

    t_end = time.perf_counter()
    logger.info(
        "add timing rid=%s chunks=%d embed=%.1fs classify+store=%.1fs facts=%.1fs "
        "summary+bm25=%.1fs gov=%.1fs total=%.1fs degraded=%s",
        req.request_id,
        n_chunks,
        t_embed - t0,
        t_store - t_embed,
        t_facts - t_store,
        t_bm25 - t_facts,
        t_end - t_bm25,
        t_end - t0,
        degraded,
    )
    _log_add_output(
        request,
        req,
        (time.perf_counter() - t0) * 1000,
        "ok-degraded" if degraded else "ok",
        n_chunks=n_chunks,
        memory_ids=memory_ids,
    )
    return AddResponse(
        success=True,
        request_id=req.request_id,
        user_id=req.user_id,
        session_id=req.session_id,
    )


@app.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(verify_api_key)],
)
async def search_memory(req: SearchRequest, request: Request) -> SearchResponse:
    t0 = time.perf_counter()
    _log_search_input(request, req)
    # claim 验证题的模板指令头不参与召回/精排：整段 query 会把 embedding 和 BM25
    # 撞到同一条 "Corpus: sentence_xxx" 模板块上，导致所有 claim 返回同一条无关记忆。
    search_query = _strip_query_boilerplate(req.query)

    # 0) 摘要/全局模式：不走向量检索，直接返回该用户全部记忆（分块优先 + 时间线）。
    #    这类请求要的是"全量覆盖"而不是 top-k 相关性，走常规链路只能覆盖一部分历史，
    #    且 [原文] 附件会把上下文膨胀数倍。失败时降级为常规检索。
    if _is_summary_request(req):
        try:
            summary_items = _build_summary_items(req)
        except Exception as exc:  # noqa: BLE001
            logger.warning("summary mode failed: %s", type(exc).__name__)
            summary_items = []
        if summary_items:
            _log_search_output(
                request,
                req,
                (time.perf_counter() - t0) * 1000,
                "ok-summary",
                summary_items,
            )
            return SearchResponse(data=summary_items)

    # 1) 并行启动：Query 理解（链路 3）+ query/选项向量化（无依赖，并行省时）
    analysis_task = asyncio.create_task(
        query_analyzer.analyze_query(req.query, req.options)
    )

    # 主 query 与 MCQ 选项级向量合并为一次批量请求（链路 4），减少一次 API 往返：
    # texts[0] 为主 query，texts[1:] 与 req.options[:10] 一一对应；无 options 时仅 [search_query]
    _opts = req.options or []
    texts = [search_query] + [f"{search_query} {opt}" for opt in _opts[:10]]
    embed_task = asyncio.create_task(embedder.embed_texts(texts))

    # 等待向量化完成。embedding 失败不再直接 500：
    # 本地 bge-m3 是 CPU 服务，机器被其它任务打满时会超时；此时降级为
    # BM25-only 检索（召回变窄但仍有结果），避免整条 /search 超时（历史故障 900s -> 504）。
    query_vector: list[float] | None = None
    option_vectors: list[list[float]] = []
    try:
        vectors = await embed_task
        query_vector = vectors[0]
        option_vectors = vectors[1:]
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "query embedding failed: %s; fallback to BM25-only", type(exc).__name__
        )

    # [v5 P5] 召回深度跟随请求的 top_k：官方请求固定 top_k=100，而固定召回 50/50
    # 在 500+ 条记忆的用户上只能拼出 ~70 个唯一候选，再被精排候选池裁到 40，
    # 正确答案可能在候选阶段就被丢弃（实测 525 条医疗用户，指定日期药物块从未进入返回集）。
    recall_k = max(config.VECTOR_TOP_K, min(req.top_k, config.RECALL_DEPTH_MAX))

    # 2) 向量召回：MCQ 选项级 + 主 query（embedding 降级时跳过，仅用 BM25）
    vector_hits: dict[str, dict] = {}
    try:
        if query_vector is not None:
            for vec in option_vectors:
                for hit in store.search_memories(
                    user_id=req.user_id, query_vector=vec, top_k=20
                ):
                    existing = vector_hits.get(hit["id"])
                    if existing is None or (hit["score"] or 0) > (existing["score"] or 0):
                        vector_hits[hit["id"]] = hit
            # 主 query 向量召回 top-50
            for hit in store.search_memories(
                user_id=req.user_id, query_vector=query_vector, top_k=recall_k
            ):
                existing = vector_hits.get(hit["id"])
                if existing is None or (hit["score"] or 0) > (existing["score"] or 0):
                    vector_hits[hit["id"]] = hit
    except Exception as exc:  # noqa: BLE001
        logger.error("vector search failed: %s", type(exc).__name__)
        _log_search_output(
            request,
            req,
            (time.perf_counter() - t0) * 1000,
            f"error:vector_search:{type(exc).__name__}",
            [],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="memory store unavailable",
        ) from exc

    # 3) BM25 召回 top-50（链路 2）
    #    [Option BM25] 主 query + 每个 MCQ 选项文本分别召回后按 memory_id 取最大分合并。
    #    正确选项常与源记忆共享字面词（如地点/物品名），对聚合/多跳型问题的召回广度提升显著；
    #    无 options 时退化为仅主 query 召回（与原有行为一致）。
    bm25_scores: dict[str, float] = {}
    for q_text in [search_query, *_opts[:10]]:
        for mid, score in bm25_index.search(req.user_id, q_text, recall_k):
            prev = bm25_scores.get(mid)
            if prev is None or score > prev:
                bm25_scores[mid] = score
    bm25_hits = sorted(
        bm25_scores.items(), key=lambda x: x[1], reverse=True
    )[:recall_k]

    t_retrieve = time.perf_counter()

    # 4) 混合召回合并：按 memory_id 去重
    candidates: dict[str, dict] = dict(vector_hits)
    if bm25_hits:
        # BM25 命中的记忆补全元数据（分数归一化到 0~1 辅助融合）
        max_bm25 = max((s for _, s in bm25_hits), default=0.0) or 1.0
        missing_ids = [mid for mid, _ in bm25_hits if mid not in candidates]
        meta_map = store.get_memories_by_ids(missing_ids) if missing_ids else {}
        for mid, bm25_score in bm25_hits:
            if mid in candidates:
                candidates[mid]["bm25_score"] = bm25_score / max_bm25
                continue
            meta = meta_map.get(mid)
            if meta is None or not meta.get("content"):
                continue
            candidates[mid] = {
                "id": mid,
                "content": meta["content"],
                "score": None,
                "bm25_score": bm25_score / max_bm25,
                "created_at": meta.get("created_at"),
                "type": meta.get("type", "other"),
                "entities": meta.get("entities", []),
                "time_expr": meta.get("time_expr"),
                "is_correction": bool(meta.get("is_correction", 0)),
                "superseded": int(meta.get("superseded", 0)),
                "importance": float(meta.get("importance", 1.0)),
                "kind": meta.get("kind", "chunk"),
            }

    candidate_list = list(candidates.values())

    # 4.2) [v3 RRF] 倒数排名融合：把 dense / BM25 两路按"在各自通道里的排名"
    #      融合，替代 v2 的 0.5*cosine + 0.5*bm25_norm 手工加权
    #      （依据：Cormack 2009 RRF；2026 lexical-dense fusion 复验）。
    #      结果写入 rrf_score，供 reranker 的无 LLM 融合分使用。
    if config.ENABLE_RRF and candidate_list:
        try:
            dense_ranked = [
                mid
                for mid, _ in sorted(
                    (
                        (mid, c.get("score"))
                        for mid, c in candidates.items()
                        if c.get("score") is not None
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            bm25_ranked = [mid for mid, _ in bm25_hits]
            rrf_norm = fusion.normalize_max(
                fusion.rrf_fuse([dense_ranked, bm25_ranked], config.RRF_K)
            )
            for c in candidate_list:
                c["rrf_score"] = rrf_norm.get(c["id"], 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rrf fusion failed: %s", type(exc).__name__)

    # 4.5) [P2] 实体关联多跳扩展（向量+BM25 合并后、治理过滤前；失败降级跳过）
    if config.ENABLE_ENTITY_EXPANSION and candidate_list:
        try:
            candidate_list = _expand_candidates_by_entities(
                req.user_id, candidate_list
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("entity expansion failed: %s", type(exc).__name__)

    # 5) 治理过滤：默认排除 superseded=1（链路 5）
    candidate_list = governance.apply_governance_filter(candidate_list)

    # query 分析结果在这里才真正需要（检索路由）。放到召回之后 await，
    # 让 LLM 延迟与向量/BM25 召回重叠；LLM 网关异常时最多只等
    # QUERY_ANALYSIS_TIMEOUT_SECONDS，不会把整条 /search 拖到 20s+。
    analysis = await analysis_task

    # 6) 检索路由（链路 3 Search 侧）
    if analysis.get("needs_latest"):
        # v3：默认软失效（同实体组旧值降权不删除），保留"变化轨迹"所需历史值
        if _is_trajectory_query(req.query):
            logger.info("trajectory query: skip soft-conflict stale penalty")
        else:
            candidate_list = governance.resolve_conflicts(
                candidate_list, needs_latest=True
            )
    qtype = analysis.get("question_type")
    if qtype == "temporal":
        # 优先含 time_expr 的记忆（排前），并按 created_at 降序
        with_time = [c for c in candidate_list if c.get("time_expr")]
        without_time = [c for c in candidate_list if not c.get("time_expr")]
        with_time.sort(key=lambda c: c.get("created_at") or "", reverse=True)
        candidate_list = with_time + without_time
    elif qtype == "preference":
        # type=preference 的记忆优先（给一个粗排权重，rerank 仍会精排）
        candidate_list.sort(key=lambda c: 0 if c.get("type") == "preference" else 1)

    # 7) LLM Rerank（链路 2；失败兜底按向量分数）
    # 多样性开启时取更大候选池，供第 9 步按 session 裁剪，避免同 session 重复项挤占名额
    global_q = _is_global_query(req.query)
    pool_k = req.top_k
    if config.DIVERSITY_PER_SESSION_CAP > 0:
        pool_k = max(req.top_k * 3, 30)
    if global_q:
        # 全局覆盖题需要跨整段历史取样，候选池开得更大
        pool_k = max(pool_k, req.top_k * 5)
    # 检索总预算兜底：/search 的尾延迟必须可控（网关拥塞时精排可挂到 20s+）。
    # 超出预算直接放弃精排，按融合分排序返回，宁可在慢窗口丢精排也不让整条请求拖长。
    _budget = config.SEARCH_TOTAL_BUDGET_SECONDS
    if _budget > 0:
        _remain = max(2.0, _budget - (time.perf_counter() - t0))
        try:
            reranked = await asyncio.wait_for(
                reranker.rerank(
                    search_query, candidate_list, top_k=pool_k, candidate_limit=recall_k
                ),
                timeout=_remain,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "search budget %.1fs exhausted before rerank finished (elapsed=%.1fs), fallback to fusion order",
                _budget,
                time.perf_counter() - t0,
            )
            reranked = sorted(
                candidate_list,
                key=lambda c: c.get("final_score")
                if c.get("final_score") is not None
                else (c.get("score") or 0.0),
                reverse=True,
            )[:pool_k]
    else:
        reranked = await reranker.rerank(
            search_query, candidate_list, top_k=pool_k, candidate_limit=recall_k
        )

    # 7.5) [P3] 弃答门控：最佳候选的精排分过低 → 判定不可回答，返回空结果。
    #      只在精排真实产出分数时生效；rerank 关闭/失败（rerank_score 为 None）时放行，
    #      避免把「精排不可用」误判成「不可回答」。
    if config.ENABLE_ABSTAIN_GATE:
        _scored = [h for h in reranked if h.get("rerank_score") is not None]
        _best_rerank = max((h["rerank_score"] for h in _scored), default=None)
        _coverage = len(_scored) / max(1, len(reranked))
        _best_aux = max((h.get("score") or 0.0 for h in reranked), default=0.0)
        # 覆盖率门槛：精排只覆盖少量候选时不足以断言「不可回答」，直接放行。
        if _best_rerank is not None and _best_rerank < config.ABSTAIN_MIN_RERANK * 2:
            logger.info(
                "gate probe: best_rerank=%.2f cover=%.2f best_cos=%.3f (query=%s)",
                _best_rerank, _coverage, _best_aux, req.query[:40],
            )
        if (
            _best_rerank is not None
            and _coverage >= config.ABSTAIN_MIN_COVERAGE
            and _best_rerank < config.ABSTAIN_MIN_RERANK
        ):
            # 二次确认：网关偶发退化（如整批返回全 0）会让单次判定误杀真正可答的问题。
            # 低分时才重排一次，两次都低才弃答；重排成本只落在疑似弃答路径上。
            _retry = await reranker.rerank(
                search_query, reranked, top_k=len(reranked)
            )
            _retry_best = max(
                (h["rerank_score"] for h in _retry if h.get("rerank_score") is not None),
                default=None,
            )
            if _retry_best is not None and _retry_best >= config.ABSTAIN_MIN_RERANK:
                logger.info(
                    "abstain vetoed by retry: best %.2f -> %.2f (query=%s)",
                    _best_rerank,
                    _retry_best,
                    req.query[:40],
                )
                reranked = _retry
            else:
                logger.info(
                    "abstain: best_rerank=%.2f/%.2f < %.2f cover=%.2f best_cos=%.3f (query=%s), return empty",
                    _best_rerank,
                    -1.0 if _retry_best is None else _retry_best,
                    config.ABSTAIN_MIN_RERANK,
                    _coverage,
                    _best_aux,
                    req.query[:40],
                )
                return SearchResponse(data=[])

    # 8) importance 加权（链路 5）：最终分数 = rerank/融合分数 × importance
    reranked = governance.apply_importance_to_score(reranked)

    # 8.5) [P3] 事实优先加成：kind='fact' 的记忆加 FACT_BOOST 后重排
    if config.PREFER_FACTS:
        boosted = False
        for h in reranked:
            if h.get("kind") == "fact":
                base = h.get("final_score")
                if base is None:
                    base = h.get("score") or 0.0
                h["final_score"] = float(base) + config.FACT_BOOST
                boosted = True
        if boosted:
            reranked.sort(key=lambda x: x["final_score"], reverse=True)

    # 8.6) [Option Overlap] MCQ 选项词面重叠加成：候选内容与任一选项文本的
    #      词面覆盖率越高，越可能是回答本题的证据；用于把分散在其它 session 的
    #      聚合型证据推进 top-k（无 options 时不变）。
    if config.ENABLE_OPTION_OVERLAP and _opts:
        option_token_sets = [_overlap_tokens(o) for o in _opts[:10]]
        relevant = [os_ for os_ in option_token_sets if os_]
        if relevant:
            boosted = False
            for h in reranked:
                ratio = _option_overlap(_overlap_tokens(h.get("content") or ""), relevant)
                if ratio <= 0:
                    continue
                base = h.get("final_score")
                if base is None:
                    base = h.get("score") or 0.0
                h["final_score"] = float(base) + config.OPTION_OVERLAP_BOOST * ratio
                boosted = True
            if boosted:
                reranked.sort(key=lambda x: x["final_score"], reverse=True)

    # 8.65) [As-of Temporal Boost] "as of <date>" 类查询（如 "job title as of Sep 05, 2025"）
    #       状态语义：应以该日期及之前最近的事件为准。按正文事件时间做衰减加成，
    #       避免同日期的通用画像事实挤占真正对应时间点的证据。
    if config.ENABLE_ASOF_BOOST:
        try:
            n_asof = _apply_asof_boost(reranked, req.query)
            if n_asof:
                logger.info("as-of boost applied to %d candidates", n_asof)
        except Exception as exc:  # noqa: BLE001
            logger.warning("as-of boost failed: %s", type(exc).__name__)

    # 8.66) [Temporal Coverage Boost] 时间类提问（"什么时候/多久/间隔"）按 query 关键词
    #       覆盖率加权：把带具体时间陈述的证据顶到前面，压住句式相近的通用画像事实。
    if config.ENABLE_TEMPORAL_COVERAGE_BOOST:
        try:
            n_tcov = _apply_temporal_coverage_boost(reranked, req.query)
            if n_tcov:
                logger.info("temporal coverage boost applied to %d candidates", n_tcov)
        except Exception as exc:  # noqa: BLE001
            logger.warning("temporal coverage boost failed: %s", type(exc).__name__)

    # 8.67) [Recency Boost] 当前状态类提问（"现在/目前/最近一次"）：新值优先
    if config.ENABLE_RECENCY_BOOST:
        try:
            n_rec = _apply_recency_boost(reranked, req.query)
            if n_rec:
                logger.info("recency boost applied to %d candidates", n_rec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("recency boost failed: %s", type(exc).__name__)

    # 8.68) [v3 Date Anchor] 无年份中文日期锚定（"1月20日前后" / "2月2日那次"）
    if config.ENABLE_DATE_ANCHOR:
        try:
            n_anchor = _apply_date_anchor_boost(reranked, req.query)
            if n_anchor:
                logger.info("date anchor boost applied to %d candidates", n_anchor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("date anchor boost failed: %s", type(exc).__name__)

    # 8.7) 分数归一化：FACT_BOOST / OPTION_OVERLAP_BOOST 叠加后可能 >1，
    #      等比缩放到 [0,1]（保序），避免对外返回越界分数。
    if config.ENABLE_SCORE_NORMALIZE:
        _normalize_scores(reranked)

    # 诊断用：AMC_DEBUG_SCORES=1 时把候选分项落盘（默认不生效，不影响线上）
    if os.environ.get("AMC_DEBUG_SCORES"):
        try:
            with open("/tmp/amc_debug_scores.json", "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "rank": i,
                            "id": h.get("id"),
                            "kind": h.get("kind"),
                            "score": h.get("score"),
                            "bm25": h.get("bm25_score"),
                            "rrf": h.get("rrf_score"),
                            "rerank": h.get("rerank_score"),
                            "final": h.get("final_score"),
                            "text": (h.get("content") or "")[:70],
                        }
                        for i, h in enumerate(reranked, start=1)
                    ],
                    f,
                    ensure_ascii=False,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("debug dump failed: %s", type(exc).__name__)

    # 9) 最终选取：按 session 多样性裁剪到 top_k（不再单纯截断前 top_k）；
    #    全局综合题改为按时间桶轮转，保证返回结果铺满整段历史（F1）
    if global_q:
        hits = _select_temporal_coverage(reranked, req.top_k)
        logger.info("global coverage selection: %d candidates for top_k=%d", len(hits), req.top_k)
    else:
        hits = _select_diverse(reranked, req.top_k, config.DIVERSITY_PER_SESSION_CAP)

    # 9.2) [v5 P0-c] 召回保底：普通检索几乎无返回时用摘要层补齐（默认关闭）
    if config.ENABLE_RECALL_FLOOR:
        try:
            hits = _apply_recall_floor(req, hits)
        except Exception as exc:  # noqa: BLE001
            logger.warning("recall floor error: %s", type(exc).__name__)

    # 9.4) 错误 QA 反馈过滤（在线学习负样本）：score 不达标的反馈及派生事实不返回
    source_ctx = None
    if config.ENABLE_PASSAGE_INTEGRATION or config.ENABLE_BAD_FEEDBACK_FILTER:
        try:
            source_ctx = _collect_source_maps(hits)
        except Exception as exc:  # noqa: BLE001
            logger.warning("source meta lookup failed: %s", type(exc).__name__)

    if config.ENABLE_BAD_FEEDBACK_FILTER and source_ctx is not None:
        try:
            meta_map, source_map, _ = source_ctx
            before_n = len(hits)
            hits = _apply_bad_feedback_filter(
                hits, meta_map, source_map, config.BAD_FEEDBACK_SCORE_THRESHOLD
            )
            if len(hits) != before_n:
                logger.info(
                    "bad-feedback filter: %d -> %d candidates", before_n, len(hits)
                )
                source_ctx = _collect_source_maps(hits)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bad-feedback filter failed: %s", type(exc).__name__)

    # 9.5) [Passage Integration] 命中事实时附加其源分块原文（可开关，失败降级）
    if config.ENABLE_PASSAGE_INTEGRATION:
        try:
            _integrate_source_passages(hits, source_ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("passage integration failed: %s", type(exc).__name__)

    # 9.6) 返回体量预算：按分数序累计字符数，避免长尾低相关记忆污染 Answer 上下文
    char_budget = (
        config.SUMMARY_CHARS_BUDGET if global_q else config.SEARCH_TOTAL_CHARS_BUDGET
    )
    hits = _trim_by_budget(hits, char_budget, config.SEARCH_MIN_ITEMS)

    # 10) 惰性 importance 更新（失败不影响响应）
    governance.bump_importance_for_hits(hits)

    items = [SearchResultItem(**{
        "id": h["id"],
        "content": h["content"],
        "score": h.get("final_score") if h.get("final_score") is not None else h.get("score"),
        "created_at": h.get("created_at"),
    }) for h in hits]
    # 分段耗时（排查线上慢请求）：t_retrieve 之前 = embed + 向量/BM25 召回
    logger.info(
        "search timing: recall=%.2fs post=%.2fs total=%.2fs",
        t_retrieve - t0,
        time.perf_counter() - t_retrieve,
        time.perf_counter() - t0,
    )
    _log_search_output(
        request, req, (time.perf_counter() - t0) * 1000, "ok", items
    )
    return SearchResponse(data=items)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        reload=False,
    )
