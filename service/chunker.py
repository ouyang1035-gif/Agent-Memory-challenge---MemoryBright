"""语义分块器：把一次 Add 的多条 messages 按语义聚合成分块。

策略（默认 CHUNK_SPLIT_ON_SPEAKER=false）：
- 按消息顺序累加，单块目标 150~250 词（中文按字符估算约 300~500 字）
- **不**在说话人切换时切分，而是跨说话人聚合，直到累加长度超过
  CHUNK_TARGET_MAX 才切分（保证单块尽量不低于 CHUNK_TARGET_MIN）
- 聚合时把每条消息格式化为 "{role}: {content}"，块内可看出谁说的
- 时间跨度大（默认 1 小时）则强制切分
- CHUNK_SPLIT_ON_SPEAKER=true 时恢复旧行为（说话人切换即切分，内容不加前缀）
- 每块保留起始 timestamp
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)

# 句子分隔符（中英文混合）
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")

# 英文完整月名（按 1~12 月顺序），不使用 strftime("%B") 以避免受运行环境 locale 影响
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


def _format_date_prefix(ts_ms: int | None) -> str:
    """把毫秒时间戳（UTC）格式化为日期前缀。

    Returns:
        形如 "[2023-07-03 July 2023] "（末尾带一个空格）的字符串；
        ts_ms 为空时返回空字符串。
    """
    if ts_ms is None:
        return ""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    month_name = _MONTH_NAMES[dt.month - 1]
    return f"[{dt.strftime('%Y-%m-%d')} {month_name} {dt.year}] "


def _apply_date_prefix(content: str, ts_ms: int | None) -> str:
    """按 CHUNK_DATE_PREFIX 开关在内容最前面拼接日期前缀。

    开关关闭或 ts_ms 为空时原样返回 content，保证行为与改动前一致。
    """
    if not config.CHUNK_DATE_PREFIX:
        return content
    return _format_date_prefix(ts_ms) + content


def _prepare_content(content: str, ts_ms: int | None) -> str:
    """对分块内容做写入侧增强：日期前缀（只追加、不改写）。

    注：相对时间归一化（P4）仅作用于事实文本（见 main._extract_and_store_facts），
    不对分块做——以免改动 25/46 条分块的 embedding/BM25 而大幅扰动检索。
    """
    return _apply_date_prefix(content, ts_ms)


@dataclass
class Chunk:
    """分块结果：一段连续对话聚合后的文本单元。"""

    role: str  # 主要说话人（块内第一条消息的 role）
    content: str  # 聚合后的文本
    start_timestamp: int | None  # 块内第一条消息的时间戳（毫秒）
    message_count: int  # 块内包含的原始消息数


def _estimate_length(text: str) -> int:
    """估算文本长度：中文按字符计，英文按词数计，混合时取较大值。"""
    if not text:
        return 0
    # 中文/日文/韩文字符数
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    # 英文按空格分词估算
    english_words = len([w for w in text.split() if w and w.isascii()])
    return max(cjk_count, english_words)


def _split_long_content(content: str, max_len: int) -> list[str]:
    """把超长单条消息按段落/句子边界切分为多个子块。

    拆分优先级：
    1) 双换行（段落边界）
    2) 单换行（行边界）
    3) 句子边界（.!?。！？+空白）
    4) 兜底：按空白硬切到接近 max_len

    Args:
        content: 原始消息内容。
        max_len: 单块目标上限（估算词数/字符数）。

    Returns:
        切分后的子串列表。每段估算长度 <= max_len（最后一段除外，
        但不会远大于 max_len）。
    """
    if _estimate_length(content) <= max_len:
        return [content]

    parts: list[str] = []

    # 1) 双换行
    paras = [p.strip() for p in content.split("\n\n") if p.strip()]
    for p in paras:
        if _estimate_length(p) <= max_len:
            parts.append(p)
            continue
        # 2) 单换行
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        buf: list[str] = []
        buf_len = 0
        for ln in lines:
            ln_len = _estimate_length(ln)
            if _estimate_length(ln) <= max_len:
                if buf and buf_len + ln_len > max_len:
                    parts.append(" ".join(buf))
                    buf = []
                    buf_len = 0
                buf.append(ln)
                buf_len += ln_len
            else:
                # 3) 句子级
                if buf:
                    parts.append(" ".join(buf))
                    buf = []
                    buf_len = 0
                sents = _SENT_SPLIT_RE.split(ln)
                sents = [s.strip() for s in sents if s.strip()]
                sb: list[str] = []
                sb_len = 0
                for s in sents:
                    s_len = _estimate_length(s)
                    if sb and sb_len + s_len > max_len:
                        parts.append(" ".join(sb))
                        sb = [s]
                        sb_len = s_len
                    else:
                        sb.append(s)
                        sb_len += s_len
                if sb:
                    # 4) 兜底：单句仍超长时按 max_len 硬切
                    for s in sb:
                        if _estimate_length(s) <= max_len:
                            parts.append(s)
                        else:
                            # 按词 / 字符硬切
                            words = s.split()
                            wb: list[str] = []
                            wb_len = 0
                            for w in words:
                                w_len = _estimate_length(w)
                                if wb and wb_len + w_len > max_len:
                                    parts.append(" ".join(wb))
                                    wb = []
                                    wb_len = 0
                                wb.append(w)
                                wb_len += w_len
                            if wb:
                                parts.append(" ".join(wb))
        if buf:
            parts.append(" ".join(buf))

    return parts if parts else [content]


def chunk_messages(messages: list[tuple[str, str, int | None]]) -> list[Chunk]:
    """把消息列表分块。

    Args:
        messages: (role, content, timestamp_ms) 元组列表，content 已保证非空。

    Returns:
        Chunk 列表。
    """
    if not config.ENABLE_CHUNKING or not messages:
        # 早退分支（逐条消息）同样按该消息自身时间戳加日期前缀，保持行为一致
        return [
            Chunk(
                role=m[0],
                content=_prepare_content(m[1], m[2]),
                start_timestamp=m[2],
                message_count=1,
            )
            for m in messages
        ]

    chunks: list[Chunk] = []
    split_on_speaker = config.CHUNK_SPLIT_ON_SPEAKER
    current_role: str | None = None
    current_ts: int | None = None
    current_count = 0
    current_len = 0
    current_parts: list[str] = []

    for role, content, ts in messages:
        # 默认跨说话人聚合，给每条消息加 "{role}: " 前缀以保留说话人信息；
        # 开关开启时保持旧行为（单说话人块，不加前缀）
        formatted_full = content if split_on_speaker else f"{role}: {content}"
        # 单条消息超长时按段落/句子/词边界切为多个子段，每段独立参与聚合
        sub_segments = _split_long_content(formatted_full, config.CHUNK_TARGET_MAX)

        for formatted in sub_segments:
            msg_len = _estimate_length(formatted)

            # 决定是否需要切分
            need_split = False
            if current_parts:
                # 时间跨度大（强制切分，允许块短于 CHUNK_TARGET_MIN）
                if (
                    ts is not None
                    and current_ts is not None
                    and abs(ts - current_ts) > config.CHUNK_TIME_GAP_MS
                ):
                    need_split = True
                # 说话人切换（仅在开关开启时切分）
                if split_on_speaker and role != current_role:
                    need_split = True
                # 长度接近上限：累加到超过 CHUNK_TARGET_MAX 才切分
                if current_len + msg_len > config.CHUNK_TARGET_MAX:
                    need_split = True

            if need_split and current_parts:
                chunks.append(
                    Chunk(
                        role=current_role or "unknown",
                        content=_prepare_content(" ".join(current_parts), current_ts),
                        start_timestamp=current_ts,
                        message_count=current_count,
                    )
                )
                current_role = None
                current_ts = None
                current_count = 0
                current_len = 0
                current_parts = []

            if current_role is None:
                current_role = role
                current_ts = ts

            current_parts.append(formatted)
            current_len += msg_len
            current_count += 1

    # 收尾
    if current_parts:
        chunks.append(
            Chunk(
                role=current_role or "unknown",
                content=_prepare_content(" ".join(current_parts), current_ts),
                start_timestamp=current_ts,
                message_count=current_count,
            )
        )

    logger.info("chunked %d messages into %d chunks", len(messages), len(chunks))
    return chunks


def chunk_to_dict(chunk: Chunk) -> dict:
    """把 Chunk 转成便于传递的字典。"""
    return {
        "role": chunk.role,
        "content": chunk.content,
        "start_timestamp": chunk.start_timestamp,
        "message_count": chunk.message_count,
    }
