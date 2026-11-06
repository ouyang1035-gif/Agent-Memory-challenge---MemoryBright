"""写入侧原子事实抽取器（P1）。

对每个分块调用 LLM，抽取自包含的原子事实列表，输出 JSON 数组：
    [{"fact": "Caroline went to the LGBTQ support group on 7 May 2023",
      "entities": ["Caroline", "LGBTQ support group"],
      "time": "2023-05-07", "type": "event"}]

要求每条事实：
- 自包含（含主语，可脱离上下文独立理解）
- 保留专名、保留绝对时间表达与具体细节
- 语言与输入原文一致
- 独立可检索（一条只讲一个事实）

语言守卫：若输入主要为英文（ASCII 占比 > 0.8）却抽出了含中文（CJK）的 fact，
视为语言违规，追加更强指令重试 1 次，重试后仍违规的条目丢弃。

失败兜底：任何异常都返回空列表，不阻塞写入。
"""

import asyncio
import json
import logging
import re

import config
import llm

logger = logging.getLogger(__name__)

_ALLOWED_TYPES = {"fact", "event", "preference", "rule", "opinion", "other"}

# 判定英文输入的 ASCII 字符占比阈值；判定中文的 CJK 区间
_ASCII_RATIO_THRESHOLD = 0.8
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

FACT_SYSTEM_PROMPT = """You are an assistant that breaks a conversation snippet into atomic facts.
你是一个对话记忆事实抽取助手。请把输入的对话片段拆解成若干条「原子事实」，以 JSON 数组输出。

每条事实的字段：
- fact: 一句话事实，必须自包含（含主语，可脱离上下文独立理解）
- entities: 该事实涉及的关键实体列表（人名、地名、组织、产品、技术、活动、物品等），没有则输出 []
- time: 事实中的时间表达（尽量规范化，如 "2023-05-07"；没有则输出 null）
- type: 事实类型，只能是 fact | event | preference | rule | opinion | other

【语言 / LANGUAGE（最重要）】
- fact 必须使用与输入原文**完全相同**的语言：输入是英文就用英文，输入是中文就用中文。
- fact MUST be written in the SAME language as the input. English input -> English fact; Chinese input -> Chinese fact.
- 严禁翻译、严禁改写为其他语言。NEVER translate. NEVER switch languages. entities 同样保持原文语言。

【时间必须保真 / TIME FIDELITY】
- 绝对时间必须**原样保留**在 fact 中："7 May 2023" 必须保留为 "7 May 2023"。
- 禁止把绝对时间改成相对表达（"last year" / "yesterday" / "recently"）；禁止省略时间。
- Keep absolute time expressions EXACTLY as written; NEVER relativize or omit them.

【日期上下文 / DATE CONTEXT】
- 若输入文本开头出现 `[YYYY-MM-DD MonthName YYYY]` 形式的日期上下文（如 `[2023-07-03 July 2023]`），它表示该对话发生的日期，请用它把原文中的**相对时间**解析为绝对时间。
  - 例如输入含 `[2023-07-03 July 2023]` 且原文说 "this month"，则 fact 应写 "July 2023"；原文说 "yesterday"，则应解析为对应日期（如 "2 July 2023"）。
- If the input begins with a date context like `[2023-07-03 July 2023]`, use it to resolve RELATIVE time expressions into absolute ones ("this month" -> "July 2023"; "yesterday" -> the corresponding date).
- 原文里**已经写明的绝对时间仍必须原样保留**，不得改写、不得省略（见上一条 TIME FIDELITY）。
- 若输入中没有日期上下文，则保持原文的时间表达，**不得编造**具体日期。
- 不得因加入日期上下文而改变 fact 的语言：fact 仍须与原文语言一致。
- NEVER fabricate a date when no date context is present; NEVER change the language of the fact because of a date context.

【细节必须保真 / DETAIL FIDELITY】
- 保留原文中的具体名词、地点、职业、数量、型号等（如 "adoption agencies"、"transgender woman"、"Prius"）。
- 禁止过度泛化：不要把 "transgender woman" 写成 "a transition"，不要把具体物品写成泛称。
- Keep concrete details; do NOT over-generalize.

【自包含 / SELF-CONTAINED】
- 每条 fact 必须含明确主语（人名或 "User"，取原文用词），可脱离上下文理解。

【只抽有信息量的事实 / INFORMATIVE ONLY】
- 跳过寒暄、情绪性空话（如 "I felt grateful"）、纯附和（如 "I agree"），除非其中包含长期偏好或个人属性。

输出要求：
- 严格输出 JSON 数组，如 [{"fact": "...", "entities": ["..."], "time": null, "type": "fact"}]
- 每条事实只讲一件事，独立可检索；不要合并多件事
- 不要编造原文没有的信息
- 只输出 JSON，不要输出任何其他解释文字

【输出格式（严格 / 省 token）】
- 只输出**单行紧凑 JSON**：不得换行、不得缩进、数组与对象之间不得留空格。
- 每条 fact 不超过 25 个词（中文不超过 40 字），但必须保留专名与绝对时间。
- entities 最多 3 个。
- 正确写法示例（必须照此风格，全部在一行内）：
  [{"fact":"Caroline attended the LGBTQ support group on 6 May 2023","entities":["Caroline","LGBTQ support group"],"time":"2023-05-06","type":"event"},{"fact":"Melanie is applying to adoption agencies","entities":["Melanie"],"time":null,"type":"fact"}]
- 错误的写法（禁止）：每个字段单独换行、数组元素逐个换行、缩进对齐。
"""


def _default() -> list[dict]:
    return []


def _parse_facts(text: str) -> list[dict]:
    """解析 LLM 返回的 JSON 数组，做字段校验与截断。失败返回空列表。"""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        logger.warning("fact response contains no JSON array: %s", text[:200])
        return []

    try:
        arr = json.loads(match.group())
    except json.JSONDecodeError as exc:
        logger.warning("fact JSON parse failed: %s", exc)
        return []

    if not isinstance(arr, list):
        return []

    limit = config.FACT_MAX_PER_CHUNK
    if limit <= 0:
        return []

    facts: list[dict] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        fact = item.get("fact")
        if not isinstance(fact, str) or not fact.strip():
            continue
        fact = fact.strip()

        raw_entities = item.get("entities")
        if isinstance(raw_entities, list):
            entities = [
                str(e).strip()
                for e in raw_entities
                if e is not None and str(e).strip()
            ]
        else:
            entities = []

        # time 必须是字符串或 None（LLM 可能返回对象/数组/数字）
        t = item.get("time")
        if t is None:
            time_val = None
        elif isinstance(t, str):
            time_val = t.strip() or None
        elif isinstance(t, (dict, list)):
            time_val = json.dumps(t, ensure_ascii=False)
        else:
            time_val = str(t)

        typ = item.get("type")
        if typ not in _ALLOWED_TYPES:
            typ = "fact"

        facts.append(
            {"fact": fact, "entities": entities, "time": time_val, "type": typ}
        )
        if len(facts) >= limit:
            break

    return facts


def _has_cjk(text: str) -> bool:
    """文本是否含中日韩（主要为中文）字符。"""
    return bool(_CJK_PATTERN.search(text))


def _is_mostly_ascii(text: str) -> bool:
    """判断输入是否主要为英文：ASCII 字符占比 > 0.8。"""
    if not text:
        return False
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    return (ascii_count / len(text)) > _ASCII_RATIO_THRESHOLD


def _violates_language(content: str, facts: list[dict]) -> bool:
    """输入主要为英文，却抽出了含中文的 fact，则视为语言违规。"""
    if not _is_mostly_ascii(content):
        return False
    return any(_has_cjk(f["fact"]) for f in facts)


_FORCE_ENGLISH_SUFFIX = (
    "\n\n=== STRICT LANGUAGE RULE ===\n"
    "The input above is in ENGLISH. You MUST output every fact and entity in ENGLISH ONLY. "
    "Do NOT use any Chinese or CJK characters. OUTPUT MUST BE IN ENGLISH ONLY."
)


async def _call_llm(content: str, force_english: bool = False) -> str:
    """调用 LLM 抽取事实，force_english 时在 user message 末尾追加更强语言指令。"""
    user_content = content + _FORCE_ENGLISH_SUFFIX if force_english else content
    return await asyncio.wait_for(
        llm.chat(
            messages=[
                {"role": "system", "content": FACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=2048,
            tier=config.LLM_TIER_CLASSIFY,
        ),
        # 跟随全局 LLM 超时（默认 60s）。网关不可用时不会把 /add 拖到分钟级。
        timeout=config.LLM_TIMEOUT_SECONDS,
    )


async def extract_facts(content: str, raise_on_error: bool = False) -> list[dict]:
    """对单个分块抽取原子事实列表。

    Args:
        content: 分块文本。
        raise_on_error: True 时把 LLM 调用失败向上抛出（供调用方重试/补抽）。

    Returns:
        事实字典列表，每项含 fact / entities / time / type。
        默认任何失败（含开关关闭）都返回空列表，不抛出异常。
    """
    if not config.ENABLE_FACT_EXTRACTION or not content or not content.strip():
        return _default()
    if config.FACT_MAX_PER_CHUNK <= 0:
        return _default()

    try:
        response = await _call_llm(content)
        facts = _parse_facts(response)

        # 语言守卫：英文输入却抽出中文事实 -> 追加更强指令重试 1 次
        if _violates_language(content, facts):
            logger.info("fact language violation detected, retrying once")
            retry_response = await _call_llm(content, force_english=True)
            retry_facts = _parse_facts(retry_response)
            if retry_facts:
                facts = retry_facts
            # 重试后仍违规（或重试失败回退到原结果）的条目丢弃
            facts = [f for f in facts if not _has_cjk(f["fact"])]

        return facts
    except Exception as exc:  # noqa: BLE001
        logger.warning("extract facts failed: %s", type(exc).__name__)
        if raise_on_error:
            raise
        return _default()
