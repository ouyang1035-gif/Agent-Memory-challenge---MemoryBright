"""Query 理解路由器：用 qwen 解析 query，输出结构化分析结果。

输出 JSON：
- question_type: fact | preference | temporal | list | mcq | open
- time_range: 起止 ISO 或 null
- key_entities: 关键实体列表
- needs_latest: bool（是否需要最新信息，用于冲突消解）

失败兜底：分析失败时返回默认值，退化为普通检索。
"""

import asyncio
import json
import logging
import re

import config
import llm

logger = logging.getLogger(__name__)

# 规则快路径用的关键词表
_TEMPORAL_HINTS = (
    "什么时候", "何时", "什么时候", "哪天", "几号", "几月", "哪年",
    "之前", "之后", "先后", "顺序", "最近一次", "第一次", "上次",
    "when", "what time", "which year", "which month", "before", "after",
)
_LATEST_HINTS = (
    "现在", "目前", "最近", "最新", "当前", "如今", "眼下",
    "now", "current", "currently", "latest", "recent", "recently",
)
_PREFERENCE_HINTS = (
    "喜欢", "爱", "偏好", "讨厌", "不喜欢", "最爱", "推荐",
    "favorite", "prefer", "like", "love", "hate", "dislike", "recommend",
)
_LIST_HINTS = (
    "哪些", "所有", "列举", "都有什么", "分别", "几种", "几门", "几件",
    "list", "all", "which ones", "what are",
)


QUERY_ANALYSIS_SYSTEM_PROMPT = """你是一个查询分析助手。请分析用户的查询，提取以下结构化信息，以 JSON 格式输出：

- question_type: 问题类型，只能是 fact | preference | temporal | list | mcq | open
  - fact: 询问具体事实
  - preference: 询问偏好/喜好
  - temporal: 询问时间相关的事件
  - list: 要求列举
  - mcq: 选择题（会提供选项）
  - open: 开放式问题
- time_range: 时间范围，如 {"start": "2024-01-01T00:00:00Z", "end": "2024-12-31T23:59:59Z"}，没有则输出 null
- key_entities: 查询中的关键实体列表，如 ["Python", "FastAPI"]，没有则输出 []
- needs_latest: 布尔值，用户是否明确要求最新/最近的信息（如"现在用什么"、"最新的版本"）

输出要求：
- 严格输出 JSON 对象，不要输出任何其他解释文字
- 字段名必须完全匹配
"""


def _default_analysis() -> dict:
    return {
        "question_type": "open",
        "time_range": None,
        "key_entities": [],
        "needs_latest": False,
    }


# 简单 LRU 缓存：同一 query 文本的分析结果（AML Streaming 会对同一题多次 Search）
_analysis_cache: dict[str, dict] = {}
_cache_order: list[str] = []


def _cache_get(key: str) -> dict | None:
    return _analysis_cache.get(key)


def _cache_put(key: str, value: dict) -> None:
    if key in _analysis_cache:
        _analysis_cache[key] = value
        return
    _analysis_cache[key] = value
    _cache_order.append(key)
    if len(_cache_order) > config.QUERY_ANALYSIS_CACHE_SIZE:
        oldest = _cache_order.pop(0)
        _analysis_cache.pop(oldest, None)


def _parse_analysis_response(text: str) -> dict:
    """解析 LLM 分析响应，失败返回默认值。"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("query analysis response contains no JSON: %s", text[:200])
        return _default_analysis()

    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning("query analysis JSON parse failed: %s", e)
        return _default_analysis()

    qtype = obj.get("question_type")
    if qtype not in ("fact", "preference", "temporal", "list", "mcq", "open"):
        qtype = "open"

    return {
        "question_type": qtype,
        "time_range": obj.get("time_range"),
        "key_entities": obj.get("key_entities") if isinstance(obj.get("key_entities"), list) else [],
        "needs_latest": bool(obj.get("needs_latest")),
    }


def _fast_path_analysis(query: str) -> dict | None:
    """规则快路径：对特征明显的 query 直接判定，跳过 LLM 调用以提速。

    返回 None 表示无法用规则判定，需走 LLM。
    """
    q = query.lower()
    has_temporal = any(h in q for h in _TEMPORAL_HINTS)
    has_latest = any(h in q for h in _LATEST_HINTS)
    has_pref = any(h in q for h in _PREFERENCE_HINTS)
    has_list = any(h in q for h in _LIST_HINTS)

    base = _default_analysis()
    if has_latest:
        base["needs_latest"] = True
    if has_temporal:
        base["question_type"] = "temporal"
        return base
    if has_pref:
        base["question_type"] = "preference"
        return base
    if has_list:
        base["question_type"] = "list"
        return base
    if has_latest:
        # 仅 needs_latest 也可走快路径
        return base
    return None


async def analyze_query(query: str, options: list[str] | None = None) -> dict:
    """分析查询，返回结构化结果。

    优先走规则快路径（无需 LLM），无法判定时才调 LLM。
    任何失败都返回默认值。
    """
    if not config.ENABLE_QUERY_ANALYSIS:
        return _default_analysis()

    # MCQ：有选项直接判定，不调 LLM
    if options:
        base = _default_analysis()
        base["question_type"] = "mcq"
        return base

    # 规则快路径
    fast = _fast_path_analysis(query)
    if fast is not None:
        return fast

    # 缓存命中
    cached = _cache_get(query)
    if cached is not None:
        return dict(cached)

    try:
        response = await asyncio.wait_for(
            llm.chat(
                messages=[
                    {"role": "system", "content": QUERY_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=512,
                tier=config.LLM_TIER_QUERY_ANALYSIS,
            ),
            timeout=config.QUERY_ANALYSIS_TIMEOUT_SECONDS,
        )
        result = _parse_analysis_response(response)
        _cache_put(query, result)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("query analysis failed: %s", type(exc).__name__)
        return _default_analysis()
