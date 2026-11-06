"""v4 P0：抽取式会话摘要（摘要记忆层）。

为每个 session 生成一条 kind='summary' 的高密度概览，供 F1（摘要/全局综合）
类请求直接命中。刻意不调用 LLM：网关 LLM 有方差；事实本身已是高密度短句；
且不写向量，因此不会污染普通 top-k 相关性检索。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import config
import store

logger = logging.getLogger(__name__)


def _iso_to_ms(created_at: str | None) -> int | None:
    if not created_at:
        return None
    try:
        return int(
            datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp() * 1000
        )
    except ValueError:
        return None


def _select_even(items: list[str], budget: int) -> list[str]:
    """按字符预算从序列中等距抽样，尽量覆盖整段会话而不是只取开头。"""
    if not items:
        return []
    total = sum(len(x) + 1 for x in items)
    if total <= budget:
        return list(items)

    n = len(items)
    avg = max(1, total // n)
    keep = max(1, min(n, budget // avg))
    if keep >= n:
        picked_idx = list(range(n))
    elif keep == 1:
        picked_idx = [0]
    else:
        picked_idx = sorted({round(i * (n - 1) / (keep - 1)) for i in range(keep)})

    picked = [items[i] for i in picked_idx]
    while picked and sum(len(x) + 1 for x in picked) > budget:
        picked.pop()
    return picked


def build_digest(rows: list[dict], max_chars: int) -> tuple[str, int | None] | None:
    """把一次 session 的记忆压成一条概览；无可用内容时返回 None。"""
    if not rows:
        return None

    facts = [r for r in rows if r.get("kind") == "fact"]
    chunks = [r for r in rows if r.get("kind") == "chunk"]

    # 事实优先（信息密度最高）；没有事实时退化为 chunk 开头片段
    if facts:
        seen: set[str] = set()
        pool: list[str] = []
        for r in facts:
            text = (r.get("content") or "").strip()
            if text and text not in seen:
                seen.add(text)
                pool.append(text)
        body_items = _select_even(pool, max(0, max_chars - 16))
    else:
        body_items = []
        for r in chunks:
            text = (r.get("content") or "").strip()
            if text:
                body_items.append(text[: max(0, max_chars - 16)])
                break

    if not body_items:
        return None

    day = (rows[0].get("created_at") or "")[:10]
    prefix = f"[{day}] " if day else ""
    digest = prefix + " ".join(body_items)
    if len(digest) > max_chars:
        digest = digest[:max_chars]

    return digest, _iso_to_ms(rows[-1].get("created_at"))


def rebuild_session_summary(user_id: str, session_id: str) -> str | None:
    """重建某 session 的摘要条目（旧摘要标记为 superseded）。失败返回 None。

    调用方负责异常兜底：摘要失败绝不能影响 /add 主链路。
    """
    if not config.ENABLE_SUMMARY_LAYER:
        return None

    rows = store.get_memories_by_session(user_id, session_id)
    n_chunks = sum(1 for r in rows if r.get("kind") == "chunk")
    if n_chunks < config.SUMMARY_SESSION_MIN_CHUNKS:
        return None

    built = build_digest(rows, config.SUMMARY_MAX_CHARS)
    if built is None:
        return None
    digest, ts_ms = built

    old = [r for r in rows if r.get("kind") == "summary"]
    if len(old) == 1 and (old[0].get("content") or "") == digest:
        return digest

    if old:
        store.mark_superseded([r["id"] for r in old])

    store.save_memories(
        request_id=f"summary-{session_id[:12]}-{int(time.time() * 1000)}",
        user_id=user_id,
        session_id=session_id,
        messages=[("summary", digest, ts_ms)],
        vectors=None,  # 只写 SQLite，不写 Qdrant / 不进 BM25
        kinds=["summary"],
    )
    logger.info(
        "session summary rebuilt: user=%s session=%s chars=%d",
        user_id[:12],
        session_id[:12],
        len(digest),
    )
    return digest
