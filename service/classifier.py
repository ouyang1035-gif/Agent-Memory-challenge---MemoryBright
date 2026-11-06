"""LLM 预分类器：对每个分块抽取结构化元数据。

输出 JSON：
- type: fact | event | preference | rule | opinion | other
- entities: 关键实体列表
- time_expr: 文本中的时间表达（原文，无则 null）
- is_correction: bool（用户是否在纠正之前的说法）

失败兜底：分类失败时默认 type=other，不阻塞写入。
"""

import asyncio
import json
import logging
import re

import config
import llm

logger = logging.getLogger(__name__)


CLASSIFY_SYSTEM_PROMPT = """你是一个对话记忆分析助手。请分析输入的对话片段，提取以下结构化信息，以 JSON 格式输出：

- type: 记忆类型，只能是 fact | event | preference | rule | opinion | other
- entities: 关键实体列表。要抽取得宽泛一些，包括：人名、地名、组织、产品、技术、编程语言、框架、食物、饮品、物品、爱好、活动等具体名词，如 ["Python", "FastAPI", "咖啡", "茶", "Alice"]，没有则输出 []
- time_expr: 文本中的时间表达原文，如 "上周"、"2024年1月1日"，没有则输出 null
- is_correction: 布尔值，表示用户是否在纠正之前的说法（如"不对，我说的是..."）

输出要求：
- 严格输出 JSON 对象，不要输出任何其他解释文字
- 字段名必须完全匹配
"""


def _parse_classify_response(text: str) -> dict:
    """解析 LLM 分类响应，失败返回默认值。"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("classify response contains no JSON object: %s", text[:200])
        return _default_classification()

    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning("classify JSON parse failed: %s", e)
        return _default_classification()

    entities = obj.get("entities") if isinstance(obj.get("entities"), list) else []
    entities = [str(e) for e in entities if e is not None]

    # time_expr 必须是字符串或 None（LLM 可能返回对象/数组，直接绑定 SQLite 会报错）
    te = obj.get("time_expr")
    if te is None or isinstance(te, str):
        time_expr = te
    elif isinstance(te, (dict, list)):
        time_expr = json.dumps(te, ensure_ascii=False)
    else:
        time_expr = str(te)

    return {
        "type": obj.get("type") if obj.get("type") in (
            "fact", "event", "preference", "rule", "opinion", "other"
        ) else "other",
        "entities": entities,
        "time_expr": time_expr,
        "is_correction": bool(obj.get("is_correction")),
    }


def _default_classification() -> dict:
    return {"type": "other", "entities": [], "time_expr": None, "is_correction": False}


async def classify_chunk(content: str) -> dict:
    """对单个分块进行 LLM 分类。

    Args:
        content: 分块文本。

    Returns:
        包含 type / entities / time_expr / is_correction 的字典。
        任何失败都返回默认值，不抛出异常。
    """
    if not config.ENABLE_CLASSIFICATION:
        return _default_classification()

    try:
        response = await asyncio.wait_for(
            llm.chat(
                messages=[
                    {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
                max_tokens=512,
                tier=config.LLM_TIER_CLASSIFY,
            ),
            timeout=30.0,  # Add 允许 30 分钟超时，这里单次分类给 30s
        )
        return _parse_classify_response(response)
    except Exception as exc:  # noqa: BLE001
        logger.warning("classify chunk failed: %s", type(exc).__name__)
        return _default_classification()
