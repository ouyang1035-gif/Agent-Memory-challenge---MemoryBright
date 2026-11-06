"""Embedding 客户端：调用内部 OpenAI 兼容 embeddings API（bge-m3）。

特性：异步 httpx、批量输入、超时控制、指数退避重试。
"""

import asyncio
import logging

import httpx

import config

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


async def init_client() -> None:
    """应用启动时初始化共享 httpx 客户端。"""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=config.EMBEDDING_TIMEOUT_SECONDS)


async def close_client() -> None:
    """应用关闭时释放 httpx 客户端。"""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _build_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {config.EMBEDDING_API_KEY}"
    return headers


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量获取 embedding 向量。

    Args:
        texts: 非空文本列表。

    Returns:
        与输入一一对应的向量列表（每个向量维度为 config.EMBEDDING_DIM）。

    Raises:
        RuntimeError: 客户端未初始化或所有重试均失败。
    """
    if not texts:
        return []
    if _client is None:
        raise RuntimeError("embedder client not initialized")

    payload = {"model": config.EMBEDDING_MODEL, "input": texts}
    last_error: Exception | None = None

    for attempt in range(config.EMBEDDING_MAX_RETRIES + 1):
        try:
            response = await _client.post(
                config.EMBEDDING_API_URL,
                headers=_build_headers(),
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            data = body.get("data")
            if not isinstance(data, list) or len(data) != len(texts):
                raise RuntimeError(
                    f"embedding API returned {0 if data is None else len(data)} "
                    f"vectors for {len(texts)} inputs"
                )
            # OpenAI 兼容格式可能乱序，按 index 归位
            sorted_data = sorted(data, key=lambda item: item.get("index", 0))
            embeddings: list[list[float]] = []
            for item in sorted_data:
                vector = item.get("embedding")
                if not isinstance(vector, list) or len(vector) != config.EMBEDDING_DIM:
                    raise RuntimeError("embedding vector missing or wrong dimension")
                embeddings.append(vector)
            return embeddings
        except Exception as exc:  # noqa: BLE001 - 重试逻辑统一兜底
            last_error = exc
            if attempt < config.EMBEDDING_MAX_RETRIES:
                backoff = 2 ** attempt
                logger.warning(
                    "embedding request failed (attempt %d/%d), retry in %ds: %s",
                    attempt + 1,
                    config.EMBEDDING_MAX_RETRIES + 1,
                    backoff,
                    type(exc).__name__,
                )
                await asyncio.sleep(backoff)

    raise RuntimeError(
        f"embedding API failed after {config.EMBEDDING_MAX_RETRIES + 1} attempts: "
        f"{type(last_error).__name__}"
    )


async def embed_query(text: str) -> list[float]:
    """单条查询文本的 embedding。"""
    vectors = await embed_texts([text])
    return vectors[0]
