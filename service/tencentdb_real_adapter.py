#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真 TencentDB-Agent-Memory (MemoryCore v3) 的比赛协议适配层。

把比赛的 Add/Search 协议映射到 MemoryCore 的 v3 RPC API：
- POST /add    -> /v3/conversation/add（写 L0，异步触发 L1 抽取）
- POST /search -> /v3/atomic/search (L1 事实) + /v3/conversation/search (L0 原文) 合并

端口 8003，与自研 (8000) / mem0 (8001) / 比赛版 tencentdb (8002) 并列。
"""
import os
import sys
import time
import logging
from typing import Any, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tcdb-adapter")

CORE_URL = os.environ.get("TDAI_CORE_URL", "http://127.0.0.1:8420")
SERVICE_ID = os.environ.get("TDAI_SERVICE_ID", "aml-bench")
PORT = int(os.environ.get("ADAPTER_PORT", "8003"))

app = FastAPI(title="tencentdb-real-adapter")


# ---------- 比赛协议模型 ----------
class ChatMessage(BaseModel):
    role: str
    content: str


class AddRequest(BaseModel):
    request_id: str
    user_id: str
    session_id: str
    messages: list[ChatMessage]
    timestamp: Optional[str] = None


class AddResponse(BaseModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    request_id: Optional[str] = None
    user_id: str
    query: str
    top_k: int = 10
    options: Optional[list[str]] = None


class SearchResultItem(BaseModel):
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    data: list[SearchResultItem]


# ---------- MemoryCore v3 客户端 ----------
def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        # standalone 模式只校验 header 存在性，token 值不校验
        "Authorization": "Bearer local-dev",
        "x-tdai-service-id": SERVICE_ID,
    }


def _isolation(user_id: str) -> dict[str, str]:
    # v3 强制 team+agent+user 三字段，缺省回落 default 桶；这里全部显式给出
    return {"team_id": "aml", "agent_id": "bench", "user_id": user_id}


async def core_add(client: httpx.AsyncClient, req: AddRequest) -> bool:
    msgs = []
    for m in req.messages:
        # MemoryCore 只接受 user|assistant；system 等其他角色归为 user
        role = m.role if m.role in ("user", "assistant") else "user"
        item: dict[str, Any] = {"role": role, "content": m.content[:8000]}
        if req.timestamp:
            item["timestamp"] = req.timestamp
        msgs.append(item)
    body = {"session_id": req.session_id, "messages": msgs, **_isolation(req.user_id)}
    r = await client.post(f"{CORE_URL}/v3/conversation/add", json=body, headers=_headers(), timeout=60)
    if r.status_code != 200:
        logger.warning("conversation/add -> %s: %s", r.status_code, r.text[:300])
        return False
    data = r.json()
    if data.get("code") != 0:
        logger.warning("conversation/add code=%s msg=%s", data.get("code"), data.get("message"))
        return False
    return True


async def core_search(client: httpx.AsyncClient, user_id: str, query: str, top_k: int) -> list[SearchResultItem]:
    iso = _isolation(user_id)
    out: list[SearchResultItem] = []

    # L1 记忆原子（LLM 异步抽取的事实，可能尚未就绪）
    try:
        r = await client.post(
            f"{CORE_URL}/v3/atomic/search",
            json={"query": query[:2000], "limit": top_k, **iso},
            headers=_headers(), timeout=30,
        )
        if r.status_code == 200 and r.json().get("code") == 0:
            for it in r.json().get("data", {}).get("items", []):
                out.append(SearchResultItem(
                    content=it.get("content", ""),
                    score=float(it.get("score", 0.0)),
                    metadata={"layer": "L1", "type": it.get("type", ""), "id": it.get("id", "")},
                ))
    except Exception as e:
        logger.warning("atomic/search error: %s", e)

    # L0 原始对话（写入即可搜，作为保底）
    try:
        r = await client.post(
            f"{CORE_URL}/v3/conversation/search",
            json={"query": query[:2000], "limit": top_k, **iso},
            headers=_headers(), timeout=30,
        )
        if r.status_code == 200 and r.json().get("code") == 0:
            for m in r.json().get("data", {}).get("messages", []):
                out.append(SearchResultItem(
                    content=m.get("content", ""),
                    score=float(m.get("score", 0.0)),
                    metadata={"layer": "L0", "session_id": m.get("session_id", ""), "id": m.get("id", "")},
                ))
    except Exception as e:
        logger.warning("conversation/search error: %s", e)

    # 按 score 降序合并，截断到 top_k；内容去重
    seen: set[str] = set()
    merged: list[SearchResultItem] = []
    for item in sorted(out, key=lambda x: x.score, reverse=True):
        key = item.content[:200]
        if key in seen or not item.content.strip():
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= top_k:
            break
    return merged


# ---------- 比赛协议端点 ----------
@app.get("/health")
def health():
    return {"status": "ok", "backend": "TencentDB-Agent-Memory MemoryCore", "core": CORE_URL}


@app.post("/add", response_model=AddResponse)
async def add(req: AddRequest):
    t0 = time.time()
    ok = False
    try:
        async with httpx.AsyncClient() as client:
            ok = await core_add(client, req)
    except Exception as e:
        logger.exception("add failed: %s", e)
    logger.info("add user=%s session=%s msgs=%d ok=%s %.2fs",
                req.user_id, req.session_id, len(req.messages), ok, time.time() - t0)
    return AddResponse(success=ok, request_id=req.request_id, user_id=req.user_id, session_id=req.session_id)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    t0 = time.time()
    results: list[SearchResultItem] = []
    try:
        async with httpx.AsyncClient() as client:
            results = await core_search(client, req.user_id, req.query, req.top_k)
    except Exception as e:
        logger.exception("search failed: %s", e)
    logger.info("search user=%s q=%r n=%d %.2fs", req.user_id, req.query[:60], len(results), time.time() - t0)
    return SearchResponse(data=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
