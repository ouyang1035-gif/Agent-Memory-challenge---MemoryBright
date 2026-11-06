"""v2 写入侧确定性「相对时间 → 绝对日期」派生事实。

背景：LLM 事实抽取（fact_extractor）倾向把相对时间原样保留（"半个月前"），
而这类表达在检索时无法落地到具体日期，直接导致 C1（日期/相对时间/间隔）
类问题失分。本模块用**纯规则**（无 LLM、无网络）把分块里的相对时间
按分块时间戳解析成绝对日期，生成一条独立可检索的事实：

    原文（user 段）: 最近一次大概是在半个月前，结果当时好像是在2点多
    派生事实:        用户自述（对话日期 2024-01-17）：……最近一次大概是在
                     半个月前，结果当时好像在2点多（「半个月前」≈ 2024-01-02）

设计原则：
- **只追加、不改写**：派生事实是新条目，原 chunk 一字不动；
- **确定性**：同一输入必然得到同一输出，便于离线回填与线上写入共用；
- **保守**：只处理有明确锚点（分块时间戳）的相对表达；跳过疑问句、
  第三人称假设句；每块最多 FACT_DERIVE_MAX_PER_CHUNK 条。

离线回填（进程内不要与服务同时跑，Qdrant 本地模式有文件锁）：

    python3 fact_derive.py --backfill [--user-id U] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sqlite3
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

# 分块首行日期前缀（chunker._format_date_prefix 产出）
_DATE_PREFIX_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} [A-Za-z]+ \d{4}\]\s*")

# 说话人分段："assistant: xxx user: yyy"（chunker 默认跨说话人聚合）
_SPEAKER_RE = re.compile(r"(?:^|\s)(user|assistant|system)\s*:\s*", re.IGNORECASE)

# 句子切分（中英文标点）
_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")

# 疑问句特征（派生事实若来自问句只会引入噪声）
_QUESTION_HINT_RE = re.compile(
    r"(吗|呢|么|多少|几点|多久|什么时候|哪一?天|是不是|要不要|"
    r"(?<![我他她])还是|有没有|会不会)"
)

# ---------- 固定中文相对表达 ----------
_CN_FIXED: dict[str, tuple[str, int]] = {
    # expr: (kind, value)，kind=day/week/month/year
    "今天": ("day", 0),
    "今日": ("day", 0),
    "今晚": ("day", 0),
    "今早": ("day", 0),
    "明天": ("day", 1),
    "明日": ("day", 1),
    "明早": ("day", 1),
    "昨天": ("day", -1),
    "昨日": ("day", -1),
    "昨晚": ("day", -1),
    "前天": ("day", -2),
    "后天": ("day", 2),
    "上周": ("week", -1),
    "上个星期": ("week", -1),
    "上星期": ("week", -1),
    "上礼拜": ("week", -1),
    "上上周": ("week", -2),
    "这周": ("week", 0),
    "本周": ("week", 0),
    "这星期": ("week", 0),
    "下周": ("week", 1),
    "下个星期": ("week", 1),
    "下星期": ("week", 1),
    "下礼拜": ("week", 1),
    "上个月": ("month", -1),
    "上上月": ("month", -2),
    "这个月": ("month", 0),
    "本月": ("month", 0),
    "下个月": ("month", 1),
    "去年": ("year", -1),
    "前年": ("year", -2),
    "今年": ("year", 0),
    "明年": ("year", 1),
}

# ---------- 数量 + 单位 + 方位 的通用中文表达 ----------
_CN_NUM_CHARS = "0-9零〇一二两三四五六七八九十百半"
_CN_REL_RE = re.compile(
    r"(?P<num>[" + _CN_NUM_CHARS + r"]{1,4})?\s*"
    r"(?P<unit>个?星期|个?礼拜|个月|天|日|周|年|小时|钟头)\s*"
    r"(?P<dir>以前|之前|以后|之后|左右|前|后|内)"
)

# ---------- 英文相对表达 ----------
_EN_FIXED: dict[str, tuple[str, int]] = {
    "today": ("day", 0),
    "tonight": ("day", 0),
    "yesterday": ("day", -1),
    "last night": ("day", -1),
    "tomorrow": ("day", 1),
    "last week": ("week", -1),
    "this week": ("week", 0),
    "next week": ("week", 1),
    "last month": ("month", -1),
    "this month": ("month", 0),
    "next month": ("month", 1),
    "last year": ("year", -1),
    "this year": ("year", 0),
    "next year": ("year", 1),
}

_EN_AGO_RE = re.compile(
    r"\b(?P<num>\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s+(?P<unit>day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)

_EN_NUM_WORDS = {
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

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

# 频率表达（"一个月左右复查一次"）→ 不当作时间点，而是间隔
_INTERVAL_HINT_RE = re.compile(r"(一次|每|频率|间隔|定期)")


def _cn_to_number(raw: str) -> float | None:
    """中文数字（含"半"/"两"/"十一"/"二十"）转 float；失败返回 None。"""
    if not raw:
        return None
    if raw.isdigit():
        return float(raw)
    if raw == "半":
        return 0.5
    if raw.count("半") == 1 and raw != "半":
        # "一个半" 之类：退化为前半部分
        head = raw.replace("半", "").replace("个", "")
        v = _cn_to_number(head) if head else 1.0
        return None if v is None else v + 0.5

    # 连写数字（"三四天"）取首个数字，表示约数
    if len(raw) > 1 and all(ch in _CN_DIGITS for ch in raw):
        return float(_CN_DIGITS[raw[0]])

    total = 0
    section = 0
    for ch in raw:
        if ch in _CN_DIGITS:
            section = _CN_DIGITS[ch]
        elif ch == "十":
            section = (section or 1) * 10
            total += section
            section = 0
        elif ch == "百":
            section = (section or 1) * 100
            total += section
            section = 0
        else:
            return None
    return float(total + section)


def _shift_months(year: int, month: int, delta: int) -> tuple[int, int]:
    """(year, month) 平移 delta 个月。"""
    m = month - 1 + delta
    return year + m // 12, m % 12 + 1


def _resolve_fixed(anchor: datetime, kind: str, value: int) -> datetime:
    """按 day/week/month/year 偏移量解析锚点日期。"""
    if kind == "day":
        return anchor + timedelta(days=value)
    if kind == "week":
        return anchor + timedelta(weeks=value)
    if kind == "month":
        y, mn = _shift_months(anchor.year, anchor.month, value)
        day = min(anchor.day, 28)
        return anchor.replace(year=y, month=mn, day=day)
    if kind == "year":
        return anchor.replace(year=anchor.year + value)
    return anchor


def _resolve_quantity(
    anchor: datetime, unit: str, num: float, direction: str
) -> tuple[datetime | None, int | None]:
    """解析「N 个单位（前/后/左右）」。

    Returns:
        (绝对日期, 间隔天数)。间隔表达（"一个月左右复查一次"）返回 (None, 天数)。
    """
    sign = 1 if direction in ("以后", "之后", "后") else -1
    unit = unit.replace("个", "")
    if unit in ("小时", "钟头"):
        hours = num * sign
        return anchor + timedelta(hours=hours), None
    if unit in ("天", "日"):
        days = num * sign
    elif unit in ("周", "星期", "礼拜"):
        days = num * 7 * sign
    elif unit == "月":
        target = _resolve_fixed(anchor, "month", int(sign * (num if num >= 1 else 1)))
        if 0 < num < 1:
            target = anchor + timedelta(days=int(round(30 * num)) * sign)
        return target, int(round(30 * abs(num)))
    elif unit == "年":
        return _resolve_fixed(anchor, "year", int(sign * (num if num >= 1 else 1))), int(
            round(365 * abs(num))
        )
    else:
        return None, None
    return anchor + timedelta(days=days), int(round(abs(days)))


def _iter_matches(sentence: str) -> list[tuple[str, tuple, bool, int]]:
    """抽取一句话里的相对时间匹配。

    Returns:
        [(原文表达式, 中间表示, 是否间隔型, 优先级)]；优先级越大越值得保留
        （3="N 个单位前/以前" 这类点位信息最强，1="今天/今晚" 这类最弱）。
    """
    out: list[tuple[str, tuple, bool, int]] = []
    consumed: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in consumed)

    # 1) 英文 "N unit ago"
    for m in _EN_AGO_RE.finditer(sentence):
        raw = m.group("num").lower()
        n = int(raw) if raw.isdigit() else _EN_NUM_WORDS.get(raw, 1)
        unit = m.group("unit").lower()
        out.append((m.group(0), ("en_ago", n, unit), False, 3))
        consumed.append((m.start(), m.end()))

    # 2) 中文固定表达（长词优先，避免 "上个月" 被 "月" 抢先）
    for expr in sorted(_CN_FIXED, key=len, reverse=True):
        for m in re.finditer(re.escape(expr), sentence):
            if _overlaps(m.start(), m.end()):
                continue
            kind, value = _CN_FIXED[expr]
            weak = expr in ("今天", "今日", "今晚", "今早")
            out.append((m.group(0), ("fixed", kind, value), False, 1 if weak else 2))
            consumed.append((m.start(), m.end()))

    # 3) 中文通用「数量+单位+方位」
    for m in _CN_REL_RE.finditer(sentence):
        if _overlaps(m.start(), m.end()):
            continue
        num = _cn_to_number(m.group("num") or "") if m.group("num") else None
        if num is None:
            continue
        unit = m.group("unit")
        direction = m.group("dir")
        if direction == "左右" and _INTERVAL_HINT_RE.search(sentence):
            out.append((m.group(0), ("qty", unit, num, direction), True, 3))
        elif direction in ("以前", "之前", "前"):
            out.append((m.group(0), ("qty", unit, num, direction), False, 3))
        elif direction in ("以后", "之后", "后"):
            # "N 天后" 多指相对另一事件（"吃药 3 天后出现胃胀"），不是相对对话日期，
            # 按锚点直接加会把日期算错，故不派生。
            continue
        else:
            out.append((m.group(0), ("qty", unit, num, direction), False, 2))
        consumed.append((m.start(), m.end()))

    # 4) 英文固定表达
    low = sentence.lower()
    for expr in sorted(_EN_FIXED, key=len, reverse=True):
        idx = low.find(expr)
        while idx != -1:
            if not _overlaps(idx, idx + len(expr)):
                kind, value = _EN_FIXED[expr]
                out.append(
                    (sentence[idx : idx + len(expr)], ("fixed", kind, value), False, 2)
                )
                consumed.append((idx, idx + len(expr)))
            idx = low.find(expr, idx + len(expr))

    return out


def _materialize(payload: tuple, anchor: datetime) -> tuple[datetime | None, int | None]:
    """把 _iter_matches 的中间表示解析成（绝对日期, 间隔天数）。"""
    kind = payload[0]
    if kind == "fixed":
        return _resolve_fixed(anchor, payload[1], payload[2]), None
    if kind == "qty":
        return _resolve_quantity(anchor, payload[1], payload[2], payload[3])
    if kind == "en_ago":
        n, unit = payload[1], payload[2]
        if unit == "day":
            return anchor - timedelta(days=n), n
        if unit == "week":
            return anchor - timedelta(weeks=n), n * 7
        if unit == "month":
            return _resolve_fixed(anchor, "month", -n), n * 30
        return _resolve_fixed(anchor, "year", -n), n * 365
    return None, None


def _is_question(sentence: str) -> bool:
    return bool(_QUESTION_HINT_RE.search(sentence)) or "?" in sentence or "？" in sentence


def _segments(content: str) -> list[tuple[str, str]]:
    """把分块切成 (role, text) 段；无说话人前缀时 role=''。"""
    matches = list(_SPEAKER_RE.finditer(content))
    if not matches:
        return [("", content)]
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        role = (m.group(1) or "").lower()
        text = content[start:end].strip()
        if text:
            out.append((role, text))
    return out


_ROLE_PREFIX = {"user": "用户自述", "assistant": "助手曾说", "system": "系统提示"}

# 多说话人转录（locomo 等）用 "[CAROLINE]: ..." 标注真实说话人；
# 分块的 role 是整个分块的（往往是 assistant），不能代表句内说话人。
_BRACKET_SPEAKER_RE = re.compile(r"^\[([A-Z][A-Za-z .'\-]{0,30})\]\s*:\s*")
# 时间未知的文本（戏剧/文档片段）：没有可信锚点，派生日期只会是错的。
_UNKNOWN_TIME_RE = re.compile(r"\[?\s*Time:\s*Unknown\s*\]?", re.IGNORECASE)


def derive_time_facts(
    content: str, ts_ms: int | None, max_per_chunk: int | None = None
) -> list[dict]:
    """从分块文本派生「相对时间 → 绝对日期」事实。

    Args:
        content: 分块原文（可含日期前缀与 "role: " 说话人前缀）。
        ts_ms: 分块起始时间戳（毫秒，UTC），作为相对时间的锚点。
        max_per_chunk: 本块最多派生几条；None 取 config。

    Returns:
        [{"fact", "entities", "time", "type"}]，语言与原文一致。
    """
    if not config.ENABLE_FACT_DERIVE or not content or ts_ms is None:
        return []
    limit = config.FACT_DERIVE_MAX_PER_CHUNK if max_per_chunk is None else max_per_chunk
    if limit <= 0:
        return []

    anchor = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    anchor_label = anchor.strftime("%Y-%m-%d")

    body = _DATE_PREFIX_RE.sub("", content, count=1)

    # 锚点不可信（Time: Unknown）时不派生，避免把错误日期写进记忆
    if _UNKNOWN_TIME_RE.search(body[:400]):
        return []

    # 1) 先收集所有候选（带优先级与出现顺序），再统一排序取前 limit 条，
    #    避免 "今天/昨天" 这类弱信息把 "半个月前" 这类关键点位挤掉。
    candidates: list[tuple[int, int, str]] = []
    order = 0
    # user 段优先（个人时间信息的一手来源）
    segments = sorted(_segments(body), key=lambda x: 0 if x[0] == "user" else 1)
    for role, text in segments:
        role_prefix = _ROLE_PREFIX.get(role, "对话记录")
        for sentence in _SENT_SPLIT_RE.split(text):
            sentence = sentence.strip()
            if len(sentence) < 4 or _is_question(sentence):
                continue
            # 句内显式说话人优先于分块 role
            prefix = role_prefix
            m_spk = _BRACKET_SPEAKER_RE.match(sentence)
            if m_spk:
                prefix = f"{m_spk.group(1).strip()} 曾说"
                sentence = sentence[m_spk.end() :].strip()
                if len(sentence) < 4 or _is_question(sentence):
                    continue
            for expr, payload, is_interval, priority in _iter_matches(sentence):
                abs_dt, days = _materialize(payload, anchor)
                if is_interval:
                    abs_dt = None
                if abs_dt is None and days is None:
                    continue
                if abs_dt is not None:
                    note = f"（「{expr}」≈ {abs_dt.strftime('%Y-%m-%d')}）"
                    time_val = abs_dt.strftime("%Y-%m-%d")
                    typ = "event"
                else:
                    note = f"（「{expr}」≈ 间隔 {days} 天）"
                    time_val = None
                    typ = "fact"
                body_text = sentence if len(sentence) <= 170 else sentence[:170] + "…"
                fact = f"{prefix}（对话日期 {anchor_label}）：{body_text}{note}"
                candidates.append(
                    (-priority, order, {"fact": fact, "entities": [], "time": time_val, "type": typ})
                )
                order += 1

    # 2) 按优先级降序、同优先级保持原文顺序，截断到 limit，去重
    candidates.sort(key=lambda x: (x[0], x[1]))
    results: list[dict] = []
    seen: set[str] = set()
    for _, _, item in candidates:
        if len(results) >= limit:
            break
        if item["fact"] in seen:
            continue
        seen.add(item["fact"])
        results.append(item)

    return results


# ---------- ② 数值/时间签名：冲突变体不合并 ----------

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_CN_NUM_IN_TEXT_RE = re.compile(r"[零〇一二两三四五六七八九十百半]+(?=天|日|周|月|年|次|个)")
_TIME_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}\s*月\s*\d{1,2}\s*[日号]")


def fact_signature(text: str) -> frozenset[str]:
    """提取事实中的数值/时间签名，用于判定"近似重复但数值冲突"。"""
    if not text:
        return frozenset()
    tokens: set[str] = set()
    tokens.update(_TIME_TOKEN_RE.findall(text))
    tokens.update(_NUM_RE.findall(text))
    for raw in _CN_NUM_IN_TEXT_RE.findall(text):
        n = _cn_to_number(raw)
        if n is not None:
            tokens.add(str(n))
    return frozenset(tokens)


def signatures_conflict(a: str, b: str) -> bool:
    """两条近似重复的事实，数值签名不同 → 视为冲突变体，应都保留。

    任一方签名为空（纯文字事实）时不判冲突，保持旧的去重行为。
    """
    if not config.FACT_DEDUP_KEEP_NUMERIC_VARIANTS:
        return False
    sa, sb = fact_signature(a), fact_signature(b)
    if not sa or not sb:
        return False
    return sa != sb


# ---------- ③ 纯口水话降权 ----------

_FILLER_RE = re.compile(
    r"^\s*(用户|user)?\s*(表示|说|觉得)?\s*(他|她|用户|我)?\s*"
    r"(同意|赞同|明白了|知道了|好的|嗯+|哦+|谢谢|感谢|收到|没问题|可以|"
    r"agree[sd]?|ok(ay)?|sounds? good|got it|thanks?|understood|sure)"
    r"[\s。.!！,，]*$",
    re.IGNORECASE,
)

_PROTECTED_TYPES = {"preference", "rule"}


def is_filler_fact(fact: str, fact_type: str | None = None) -> bool:
    """极窄判定「纯口水话」事实（不碰偏好/规则类，保护 E 类得分）。"""
    if not config.ENABLE_FILLER_IMPORTANCE_DOWNWEIGHT or not fact:
        return False
    if (fact_type or "").lower() in _PROTECTED_TYPES:
        return False
    if _NUM_RE.search(fact):
        return False
    if len(fact) > 24:
        return False
    return bool(_FILLER_RE.match(fact))


# ---------- 离线回填 ----------

async def _backfill(user_id: str | None = None, dry_run: bool = False) -> None:
    """对已有 kind='chunk' 记忆补生成派生事实（v2 首次上线前执行一次）。

    注意：Qdrant 本地模式有文件锁，回填时**不要**同时运行 8010 服务。
    """
    import embedder
    import store

    await embedder.init_client()
    store.init_store()
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    sql = "SELECT id, user_id, session_id, request_id, content, created_at, role FROM memories WHERE kind='chunk'"
    params: list = []
    if user_id:
        sql += " AND user_id = ?"
        params.append(user_id)
    rows = conn.execute(sql, params).fetchall()
    # 幂等：已经回填过派生事实的分块直接跳过，可重复执行
    done = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT source_id FROM memories "
            "WHERE kind='fact' AND request_id LIKE '%:derive'"
        ).fetchall()
    }
    conn.close()
    logger.info("backfill: scanning %d chunks (already derived: %d)", len(rows), len(done))

    total_chunks = derived_chunks = derived_facts = saved = 0
    for row in rows:
        total_chunks += 1
        if row["id"] in done:
            continue
        created = row["created_at"]
        try:
            ts_ms = int(
                datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp() * 1000
            )
        except Exception:  # noqa: BLE001
            continue
        facts = derive_time_facts(row["content"], ts_ms)
        if not facts:
            continue
        derived_chunks += 1
        derived_facts += len(facts)
        if dry_run:
            if derived_facts <= 12:
                for f in facts:
                    print(f"[{row['id'][:8]}] {f['fact'][:150]}")
            continue
        try:
            vectors = await embedder.embed_texts([f["fact"] for f in facts])
        except Exception as exc:  # noqa: BLE001
            logger.warning("backfill embed failed: %s", type(exc).__name__)
            continue
        metadata = [
            {"type": f["type"], "entities": f["entities"], "time_expr": f["time"],
             "is_correction": False}
            for f in facts
        ]
        ids = store.save_memories(
            request_id=(row["request_id"] or "backfill") + ":derive",
            user_id=row["user_id"],
            session_id=row["session_id"] or "",
            messages=[("fact", f["fact"], ts_ms) for f in facts],
            vectors=vectors,
            metadata_list=metadata,
            kinds=["fact"] * len(facts),
            source_ids=[row["id"]] * len(facts),
        )
        saved += len(ids)

    print(
        f"backfill done: chunks={total_chunks} chunks_with_derive={derived_chunks} "
        f"derived_facts={derived_facts} saved={saved} dry_run={dry_run}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="v2 相对时间派生事实")
    parser.add_argument("--backfill", action="store_true", help="对已有 chunk 回填")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.backfill:
        asyncio.run(_backfill(user_id=args.user_id, dry_run=args.dry_run))
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
