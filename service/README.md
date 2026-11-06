# Agent Memory Service — 链路 1（最小可行记忆服务）

面向 Agent Memory Challenge 2026（AML 平台评测）的 HTTP 记忆服务。
链路 1 为纯向量检索版本：无分块策略（按消息粒度）、无分类、无 rerank。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/add` | 写入记忆（幂等，按 request_id 去重） |
| POST | `/search` | 按 user_id 隔离的向量检索 |
| GET | `/health` | 健康检查（无鉴权） |

## 架构

- **FastAPI + Uvicorn**：HTTP 服务
- **Qdrant**（Docker，localhost:6333）：向量存储，collection `memories`，1024 维 cosine
- **SQLite**（`./memory.db`）：元数据 + 幂等去重（request_id 索引）
- **bge-m3 Embedding**：内部 OpenAI 兼容 API，异步批量调用，重试 2 次

写入流程：幂等检查（SQLite）→ 批量 embedding → 写 SQLite → 写 Qdrant（wait=true 同步落盘）→ 返回 200。

## 启动步骤

### 1. 安装依赖

```bash
cd memory-service
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 启动 Qdrant（Docker）

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### 3. 配置环境变量

```bash
# 复制模板
cp .env.example .env        # Windows: copy .env.example .env
```

编辑 `.env`，把 `__PUT_YOUR_EMBEDDING_API_KEY_HERE__` 替换为你自己的 Embedding API Key。
**Key 严禁提交到版本库**（`.env` 已在 `.gitignore` 中）。

可选：设置 `SERVICE_API_KEY` 后，`/add` 和 `/search` 需要携带 `X-Api-Key` 头；留空则放行。

### 4. 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# 或：
python main.py
```

启动时会自动创建 Qdrant collection 和 SQLite 表。

### 5. curl 自测

以下示例中的 `__YOUR_SERVICE_API_KEY__` 为占位符：若 `.env` 未配置 `SERVICE_API_KEY`，可去掉 `X-Api-Key` 头；若配置了，替换为你的 Key。

```bash
# 健康检查
curl http://localhost:8000/health

# 写入记忆
curl -X POST http://localhost:8000/add \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: __YOUR_SERVICE_API_KEY__" \
  -d '{
    "request_id": "test:run_001:conv-0:chunk-0",
    "messages": [
      {"role": "user", "timestamp": 1704067200000, "content": "我喜欢喝咖啡，尤其是手冲。"},
      {"role": "assistant", "timestamp": 1704067205000, "content": "手冲咖啡确实能更好地保留风味。"}
    ],
    "user_id": "test:run_001:conv-0",
    "session_id": "test:run_001:sample:0"
  }'

# 检索记忆
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: __YOUR_SERVICE_API_KEY__" \
  -d '{
    "query": "用户喜欢什么饮品？",
    "user_id": "test:run_001:conv-0",
    "top_k": 5
  }'
```

预期 `/search` 响应：

```json
{
  "data": [
    {
      "id": "<uuid4-hex>",
      "content": "我喜欢喝咖啡，尤其是手冲。",
      "score": 0.83,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

重复发送相同 `request_id` 的 `/add` 请求会返回 200 且原样返回三个 ID，不会重复写入。

## 注意事项

- **user_id 隔离是合规红线**：检索强制以 `user_id` 作为 Qdrant filter，跨用户不可见。
- 日志不记录请求正文全文，仅记录 request_id / user_id / 条数等概要信息。
- 内部错误统一返回 500 + 通用文案，不泄露堆栈与配置细节。
- `llm.py` 为链路 2+ 预留，链路 1 不调用。
