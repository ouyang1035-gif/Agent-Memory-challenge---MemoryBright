"""写入侧确定性相对时间归一化（P4）。

把英文相对时间表达式按会话日期（session timestamp）解析为绝对时间，
并以括号注记的形式**追加**在原文之后，绝不改写原文：

    "Jolene got Seraphim last year."
        -> "Jolene got Seraphim last year (2022)."

设计原则：
- **只追加、不改写**：原文完整保留（embedding 仍能匹配原表达），
  仅额外补充绝对时间 token，便于按 "2022" 之类的绝对时间来检索。
- **确定性**：纯规则解析，不依赖 LLM；无法解析时原样返回。
- **幂等**：若原文该表达式后已紧跟括号注记，则不再重复追加。
- 表达式已在括号内（如 "(last year)"）时不处理，避免误伤。

覆盖的英文相对时间：yesterday / today / tomorrow / tonight / last night、
last|this|next week|month|year、last|next|this <weekday>、
以及 "<N> <day|week|month|year>(s) ago"（N 支持数字或英文数词）。
"""

import logging
import re
from datetime import datetime, timedelta, timezone

import config

logger = logging.getLogger(__name__)

_MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_WEEKDAYS: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_NUM_WORDS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_WEEKDAY_ALT = "|".join(_WEEKDAYS)

# "<N> <unit> ago"（N 为数字或英文数词）
_AGO_RE = re.compile(
    r"\b(\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s+(day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)

# "last|this|next <weekday>"
_WEEKDAY_RE = re.compile(
    r"\b(last|next|this)\s+(" + _WEEKDAY_ALT + r")\b",
    re.IGNORECASE,
)

# "last|this|next <week|month|year>"
_UNIT_RE = re.compile(
    r"\b(last|next|this)\s+(week|month|year)\b",
    re.IGNORECASE,
)

# yesterday / today / tomorrow / tonight
_SIMPLE_RE = re.compile(
    r"\b(yesterday|today|tomorrow|tonight)\b",
    re.IGNORECASE,
)

# last night（在 _SIMPLE_RE 之外单独处理）
_LAST_NIGHT_RE = re.compile(r"\blast night\b", re.IGNORECASE)


def _shift_months(year: int, month: int, delta: int) -> tuple[int, int]:
    """在 (year, month) 上平移 delta 个月，返回新的 (year, month)。"""
    m = month - 1 + delta
    return year + m // 12, m % 12 + 1


def _has_note_after(text: str, end: int) -> bool:
    """判断 text[end:] 之后是否已紧跟括号注记（如 " (2022)"）。"""
    return text[end:].lstrip()[:1] == "("


def _sub_with_guard(pattern: re.Pattern, text: str, repl) -> str:
    """执行 re.sub，但跳过其后已带括号注记的匹配（幂等保护）。"""

    def _wrapped(m: re.Match) -> str:
        if _has_note_after(text, m.end()):
            return m.group(0)
        return repl(m)

    return pattern.sub(_wrapped, text)


def normalize_time(text: str, ts_ms: int | None) -> str:
    """把 text 中的英文相对时间解析为绝对时间并以括号注记追加。

    Args:
        text: 待归一化文本（可为分块内容或单条事实）。
        ts_ms: 会话/分块起始时间戳（毫秒，UTC）。

    Returns:
        归一化后的文本；开关关闭、ts 为空或文本为空时原样返回。
    """
    if not config.ENABLE_TIME_NORMALIZE or not text or ts_ms is None:
        return text

    base = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

    # 1) "<N> <unit> ago"
    def _repl_ago(m: re.Match) -> str:
        raw = m.group(1).lower()
        n = int(raw) if raw.isdigit() else _NUM_WORDS.get(raw, 1)
        unit = m.group(2).lower()
        if unit == "day":
            label = (base - timedelta(days=n)).strftime("%Y-%m-%d")
        elif unit == "week":
            label = (base - timedelta(weeks=n)).strftime("%Y-%m-%d")
        elif unit == "month":
            y, mn = _shift_months(base.year, base.month, -n)
            label = f"{_MONTH_NAMES[mn - 1]} {y}"
        else:  # year
            label = str(base.year - n)
        return f"{m.group(0)} ({label})"

    # 2) "last|this|next <week|month|year>"
    def _repl_unit(m: re.Match) -> str:
        kind = m.group(1).lower()
        unit = m.group(2).lower()
        delta = -1 if kind == "last" else (1 if kind == "next" else 0)
        if unit == "week":
            label = (base + timedelta(weeks=delta)).strftime("%Y-%m-%d")
        elif unit == "month":
            y, mn = _shift_months(base.year, base.month, delta)
            label = f"{_MONTH_NAMES[mn - 1]} {y}"
        else:  # year
            label = str(base.year + delta)
        return f"{m.group(0)} ({label})"

    # 3) "last|next|this <weekday>"
    def _repl_weekday(m: re.Match) -> str:
        kind = m.group(1).lower()
        target_idx = _WEEKDAYS[m.group(2).lower()]
        if kind == "last":
            days = ((base.weekday() - target_idx - 1) % 7) + 1
            target = base - timedelta(days=days)
        elif kind == "next":
            days = ((target_idx - base.weekday() - 1) % 7) + 1
            target = base + timedelta(days=days)
        else:  # this
            target = base + timedelta(days=target_idx - base.weekday())
        label = target.strftime("%Y-%m-%d")
        return f"{m.group(0)} ({label})"

    # 4) yesterday / today / tomorrow / tonight
    def _repl_simple(m: re.Match) -> str:
        word = m.group(1).lower()
        if word == "yesterday":
            target = base - timedelta(days=1)
        elif word == "tomorrow":
            target = base + timedelta(days=1)
        else:  # today / tonight
            target = base
        return f"{m.group(0)} ({target.strftime('%Y-%m-%d')})"

    # 5) last night -> 前一天
    def _repl_last_night(m: re.Match) -> str:
        target = base - timedelta(days=1)
        return f"{m.group(0)} ({target.strftime('%Y-%m-%d')})"

    out = text
    out = _sub_with_guard(_AGO_RE, out, _repl_ago)
    out = _sub_with_guard(_WEEKDAY_RE, out, _repl_weekday)
    out = _sub_with_guard(_UNIT_RE, out, _repl_unit)
    out = _sub_with_guard(_LAST_NIGHT_RE, out, _repl_last_night)
    out = _sub_with_guard(_SIMPLE_RE, out, _repl_simple)
    return out
