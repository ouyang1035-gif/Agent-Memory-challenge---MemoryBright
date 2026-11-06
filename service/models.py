"""Pydantic 请求/响应模型，严格对齐 AML 平台接口契约。"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------- POST /add ----------


class MessageItem(BaseModel):
    role: str = Field(..., min_length=1)
    content: str = Field(default="")
    timestamp: int | None = Field(default=None, description="Unix 毫秒时间戳，可选")


class AddRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    messages: list[MessageItem] = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class AddResponse(BaseModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


# ---------- POST /search ----------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    options: list[str] | None = Field(default=None, description="选择题选项，链路 1 忽略")
    user_id: str = Field(..., min_length=1)
    top_k: int = Field(default=100, ge=1, le=1000)


class SearchResultItem(BaseModel):
    id: str
    content: str
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    data: list[SearchResultItem]


# ---------- 通用错误响应 ----------


class ErrorResponse(BaseModel):
    detail: str
