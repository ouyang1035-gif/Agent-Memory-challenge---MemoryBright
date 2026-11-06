"""LLM 客户端：调用内部 OpenAI 兼容 chat completions API。

支持两个模型档位（见 config.resolve_llm）：
- small：默认档（快），用于分类、query 理解等高频简单任务
- large：大模型档（质量优先），用于 rerank、冲突治理等语义判断

大模型档未配置，或调用失败时，自动回退到默认档，保证服务可用性。
"""

import asyncio
import logging
import random
import time

import httpx

import config

logger = logging.getLogger(__name__)


# 熔断：网关不可用（超时/连接失败）时，每个请求都要白等一个完整超时。
# 2026-10-27 线上：/search 每次多 20s、/add 每次多 40s（分类 + 事实抽取各一次）。
# 连续 N 次网络级失败后暂停调用 COOLDOWN 秒，直接抛错让上层走降级；
# 冷却结束自动放一个请求探测，网关恢复即自愈。
#
# 2026-10-29 修订：阈值 3 / 冷却 300s 太激进——官方 smoke 里 27 次调用中 9 次
# 网关抖动就把熔断打开，剩下 134 次 /add 的分类与事实抽取全部秒失败，
# 记忆产量掉了 90%（A1/C1 直接崩）。改为阈值 8 / 冷却 60s，并配合重试退避。
_BREAKER_FAIL_THRESHOLD = config.LLM_BREAKER_FAIL_THRESHOLD
_BREAKER_COOLDOWN_SECONDS = config.LLM_BREAKER_COOLDOWN_SECONDS
_breaker_failures = 0
_breaker_until = 0.0
# 冷却结束后的半开探测：只放行一个请求试水，成功即彻底关闭熔断。
_breaker_probing = False


def _breaker_acquire() -> bool:
    """放行判定。True=放行，False=快速失败。

    冷却期结束后第一个到达的请求作为「半开探测」，其余继续快速失败，
    探测成功即彻底关闭熔断，失败则立刻重新打开并重新计时。
    """
    global _breaker_probing
    now = time.monotonic()
    if now < _breaker_until:
        return False
    if _breaker_until > 0.0:
        if _breaker_probing:
            return False
        _breaker_probing = True
    return True


def _breaker_record_failure() -> None:
    global _breaker_failures, _breaker_until, _breaker_probing
    _breaker_failures += 1
    if _breaker_probing or _breaker_failures >= _BREAKER_FAIL_THRESHOLD:
        _breaker_until = time.monotonic() + _BREAKER_COOLDOWN_SECONDS
        _breaker_probing = False
        logger.warning(
            "LLM circuit opened for %.0fs after %d consecutive network failures",
            _BREAKER_COOLDOWN_SECONDS,
            _breaker_failures,
        )


def _breaker_record_success() -> None:
    global _breaker_failures, _breaker_until, _breaker_probing
    _breaker_failures = 0
    _breaker_until = 0.0
    _breaker_probing = False


# 网关在过载时返回的瞬时状态码，语义上等同网络失败，必须计入熔断
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

# 网关把「模型不健康 / 限流」也塞进 400 里返回（HTTP 400 但同属瞬时故障）：
#   {"error":{"message":"当前模型不健康，请稍后重试","code":"400"}}
#   {"error":{"message":"请求频率过高，请稍后重试","code":"429"}}
# 这类必须重试，否则官方评测期间会成片丢结果。
_TRANSIENT_DETAIL_HINTS = (
    "模型不健康",
    "不健康",
    "请求频率过高",
    "频率过高",
    "限流",
    "overloaded",
    "rate limit",
    "too many requests",
)


class _TransientGatewayError(RuntimeError):
    """网关瞬时故障（模型不健康 / 限流），可重试且计入熔断。"""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


def _detail_is_transient(detail: str) -> bool:
    low = (detail or "").lower()
    return any(h in low for h in _TRANSIENT_DETAIL_HINTS)


def _is_network_error(exc: BaseException) -> bool:
    """只有网络/网关过载层面的失败才计入熔断（空 content / 解析失败不算）。"""
    if isinstance(exc, _TransientGatewayError):
        return True
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    # 2026-10-28：此前只认超时/连接错误，网关的 502/429 被当成「业务失败」不计入，
    # 结果熔断永远不触发，每个请求都白等一轮超时 + 重试后降级。
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS
    return False


# 在途请求闸门：网关并发一高就 502/429，先排队再发可避免自我加剧。
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max(1, config.LLM_MAX_CONCURRENCY))
    return _semaphore


async def _request(
    model: str,
    api_url: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> str:
    """单次底层请求，返回助手 content。"""
    if not api_key:
        raise RuntimeError("LLM API key is not configured")

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if not config.LLM_ENABLE_THINKING:
        # 推理模型（如 glm-5.2）默认会把 max_tokens 耗在 reasoning 上，
        # 导致 content 为空；显式关闭思考以保证结构化输出。
        # 注意：vLLM 只认 chat_template_kwargs 里的 enable_thinking，
        # 顶层 enable_thinking 会被它静默忽略（思考过程混进 content）。
        if config.LLM_DISABLE_THINKING_STYLE == "template":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["enable_thinking"] = False
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with _get_semaphore():
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(api_url, headers=headers, json=payload)
            if response.status_code >= 400:
                detail = ""
                try:
                    err = response.json().get("error") or {}
                    detail = str(err.get("message") or err)
                except Exception:  # noqa: BLE001
                    detail = response.text[:200]
                if (
                    response.status_code in _TRANSIENT_STATUS
                    or _detail_is_transient(detail)
                ):
                    raise _TransientGatewayError(
                        f"gateway transient HTTP {response.status_code}: {detail[:120]}",
                        response.status_code,
                    )
                response.raise_for_status()
            body = response.json()

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("LLM API returned no choices")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError("LLM API returned invalid message content")
    if not content.strip():
        # 网关偶发返回空 content；抛出以触发上层重试，避免静默丢结果。
        raise RuntimeError("LLM API returned empty content")
    return content


async def chat(
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    tier: str = "small",
) -> str:
    """调用 chat completions，返回助手消息文本。

    Args:
        messages: OpenAI 格式的消息列表，如 [{"role": "user", "content": "..."}]。
        temperature: 采样温度。
        max_tokens: 最大生成 token 数，None 表示不限制。
        tier: 模型档位，"small"（默认）或 "large"。large 未配置时自动回退 small。

    Returns:
        助手回复的 content 字符串。

    Raises:
        RuntimeError: Key 未配置或请求失败（大模型档失败会先尝试回退默认档）。
    """
    if not _breaker_acquire():
        raise RuntimeError("LLM gateway circuit open (recent network failures)")

    try:
        return await _chat_with_retries(
            messages, temperature, max_tokens, tier
        )
    except asyncio.CancelledError:
        # 上层 wait_for 超时会取消本协程（网关挂掉时 query 分析和事实抽取都是这样超时的），
        # 这同样是网关不可用的证据，必须计入熔断。
        _breaker_record_failure()
        raise


# 重试退避：0.4s / 1.0s / 2.0s（再叠加 0~0.3s 抖动，避免并发请求同时重发）
_RETRY_BACKOFF = (0.4, 1.0, 2.0)
# 限流（429）要等窗口滑过去，退避拉长；否则 1s 内重发必然再撞 429
_RETRY_BACKOFF_RATE_LIMIT = (2.0, 5.0, 10.0)


def _is_retryable(exc: BaseException) -> bool:
    """可重试：网关瞬时故障 / 网络失败 / 空 content。其余（鉴权、参数）直接抛。"""
    if _is_network_error(exc):
        return True
    msg = str(exc)
    return "empty content" in msg or "no choices" in msg


def _backoff_delay(attempt: int, exc: BaseException) -> float:
    schedule = (
        _RETRY_BACKOFF_RATE_LIMIT
        if isinstance(exc, _TransientGatewayError) and exc.status == 429
        else _RETRY_BACKOFF
    )
    base = schedule[min(attempt, len(schedule) - 1)]
    return base + random.uniform(0.0, 0.3)


async def _request_with_retries(
    model: str,
    api_url: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
    attempts: int,
) -> tuple[str, Exception | None]:
    """带退避重试的单档调用。返回 (content, None) 或 ("", last_exc)。"""
    last_exc: Exception | None = None
    started = time.monotonic()
    for attempt in range(attempts):
        try:
            content = await _request(
                model, api_url, api_key, messages, temperature, max_tokens
            )
            _breaker_record_success()
            return content, None
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_retryable(exc) or attempt == attempts - 1:
                break
            # 重试总预算：避免「超时 20s × 4 次」把 /add 拖到分钟级。
            if time.monotonic() - started >= config.LLM_RETRY_BUDGET_SECONDS:
                logger.warning(
                    "LLM retry budget exhausted after %d attempt(s)", attempt + 1
                )
                break
            delay = _backoff_delay(attempt, exc)
            logger.info(
                "LLM retry %d/%d in %.1fs after %s: %s",
                attempt + 1,
                attempts,
                delay,
                type(exc).__name__,
                str(exc)[:120],
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    return "", last_exc


async def _chat_with_retries(
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
    tier: str,
) -> str:
    model, api_url, api_key = config.resolve_llm(tier)
    attempts = max(1, config.LLM_MAX_ATTEMPTS)

    content, last_exc = await _request_with_retries(
        model, api_url, api_key, messages, temperature, max_tokens, attempts
    )
    if last_exc is None:
        return content

    # 大模型档多次失败 -> 回退默认档（若本身已是默认档则直接抛出）
    if tier == "large" and config.LLM_MODEL_LARGE:
        logger.warning(
            "large model (%s) call failed (%s), fallback to %s",
            model, type(last_exc).__name__, config.LLM_MODEL,
        )
        fb_model, fb_url, fb_key = config.resolve_llm("small")
        content, fb_exc = await _request_with_retries(
            fb_model, fb_url, fb_key, messages, temperature, max_tokens, attempts
        )
        if fb_exc is None:
            return content
        last_exc = fb_exc

    if _is_network_error(last_exc):
        _breaker_record_failure()
    raise last_exc
