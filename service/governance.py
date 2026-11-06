"""记忆治理模块（链路 5）：

- Add 侧：is_correction=true 时，用 qwen 找出被纠正的旧记忆并标记 superseded=1
- Search 侧：默认过滤 superseded=1；最终分数 = rerank_score × importance
- importance 惰性更新：被命中返回的记忆 importance +0.05 封顶 2.0
  TODO: 后续可加时间衰减定时任务
"""

import asyncio
import json
import logging
import re

import config
import llm
import store

logger = logging.getLogger(__name__)


SUPERSEDE_SYSTEM_PROMPT = """你是一个记忆治理助手。用户刚纠正了一个说法（新记忆），下面给出可能相关的旧记忆列表。
请判断哪些旧记忆的内容被这条新记忆所纠正/取代（即新旧说法矛盾，且新说法覆盖了旧说法）。

输出要求：
- 严格输出 JSON 数组，元素为被取代旧记忆的 id，如 ["id1", "id2"]
- 如果没有旧记忆被取代，输出 []
- 只输出 JSON，不要输出任何其他文字
"""


def _parse_supersede_response(text: str, valid_ids: set[str]) -> list[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        arr = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    return [mid for mid in arr if isinstance(mid, str) and mid in valid_ids]


async def resolve_correction(
    user_id: str, new_content: str, new_entities: list[str], new_memory_id: str
) -> list[str]:
    """处理纠正型新记忆：找出被纠正的旧记忆并标记 superseded。

    Returns:
        被标记 superseded 的旧记忆 id 列表。
    """
    if not config.ENABLE_GOVERNANCE:
        return []

    if new_entities:
        candidates = store.find_supersede_candidates(
            user_id=user_id, entities=new_entities, exclude_ids=[new_memory_id]
        )
    else:
        # Fallback：纠正消息未抽到实体时，取该用户全部未取代旧记忆，
        # 交给 LLM 自行判断哪些被新说法取代（避免实体抽取保守导致治理失效）。
        candidates = [
            m
            for m in store.get_memories_by_user(user_id)
            if m["id"] != new_memory_id and not m.get("superseded")
        ]
    if not candidates:
        return []

    lines = [f"新记忆（纠正说法）：{new_content}", "", "可能相关的旧记忆："]
    for c in candidates:
        lines.append(f"[{c['id']}] {c['content']}")
    lines.append("")
    lines.append("请输出被新记忆取代的旧记忆 id 的 JSON 数组：")

    valid_ids = {c["id"] for c in candidates}
    try:
        response = await asyncio.wait_for(
            llm.chat(
                messages=[
                    {"role": "system", "content": SUPERSEDE_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                temperature=0.0,
                max_tokens=512,
                tier=config.LLM_TIER_GOVERNANCE,
            ),
            timeout=30.0,
        )
        to_supersede = _parse_supersede_response(response, valid_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("supersede resolution failed: %s", type(exc).__name__)
        return []

    if to_supersede:
        n = store.mark_superseded(to_supersede)
        logger.info("superseded %d memories for user_id=%s", n, user_id)
    return to_supersede


def apply_governance_filter(candidates: list[dict]) -> list[dict]:
    """Search 侧过滤：默认排除 superseded=1 的记忆。"""
    if not config.ENABLE_GOVERNANCE:
        return candidates
    return [c for c in candidates if not c.get("superseded")]


def apply_importance_to_score(candidates: list[dict]) -> list[dict]:
    """Search 侧打分：最终分数 = 融合分数 × importance × 软失效惩罚。"""
    if not config.ENABLE_GOVERNANCE:
        return candidates
    for c in candidates:
        importance = float(c.get("importance") or 1.0)
        base = c.get("final_score")
        if base is None:
            base = c.get("score") or 0.0
        c["final_score"] = base * importance * float(c.get("stale_penalty") or 1.0)
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates


def bump_importance_for_hits(hits: list[dict]) -> None:
    """惰性 importance 更新：被返回的记忆 importance +delta（封顶 cap）。"""
    if not config.ENABLE_GOVERNANCE or not hits:
        return
    try:
        store.bump_importance(
            [h["id"] for h in hits],
            delta=config.GOVERNANCE_IMPORTANCE_BOOST,
            cap=config.GOVERNANCE_IMPORTANCE_CAP,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("bump importance failed: %s", type(exc).__name__)


def resolve_conflicts_latest(candidates: list[dict], needs_latest: bool) -> list[dict]:
    """冲突消解：needs_latest=true 时，entities 高度重叠的记忆只保留 created_at 最新的。

    重叠判定：entities 集合交集 / 并集 ≥ 0.5 视为高度重叠。
    """
    if not needs_latest or not candidates:
        return candidates

    kept: list[dict] = []
    # 按 created_at 降序遍历，新者优先保留
    sorted_cands = sorted(
        candidates, key=lambda x: x.get("created_at") or "", reverse=True
    )
    for c in sorted_cands:
        c_entities = set(c.get("entities") or [])
        conflict = False
        for k in kept:
            if _overlap(c_entities, set(k.get("entities") or [])) >= 0.5:
                conflict = True
                break
        if not conflict:
            kept.append(c)
    return kept


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _soft_conflict(candidates: list[dict]) -> list[dict]:
    """软失效：同实体组内非最新的记忆只降权（stale_penalty），不删除。

    与 Zep 的 edge invalidation / Mem0 的 soft re-rank 一致：保留历史值，
    让"变化轨迹"类问题仍能取到旧值；只保证当前状态类问题里新值排在前面。
    """
    penalty = float(config.CONFLICT_STALE_PENALTY)
    reps: list[dict] = []
    for c in sorted(candidates, key=lambda x: x.get("created_at") or "", reverse=True):
        c_entities = set(c.get("entities") or [])
        conflicted = any(
            _overlap(c_entities, set(r.get("entities") or [])) >= 0.5 for r in reps
        )
        if conflicted:
            c["stale_penalty"] = penalty
        else:
            reps.append(c)
    return candidates


def resolve_conflicts(candidates: list[dict], needs_latest: bool) -> list[dict]:
    """needs_latest 时的冲突消解入口：默认软失效，可回退为旧版硬删。"""
    if not needs_latest or not candidates:
        return candidates
    if config.ENABLE_SOFT_CONFLICT:
        return _soft_conflict(candidates)
    return resolve_conflicts_latest(candidates, needs_latest=True)
