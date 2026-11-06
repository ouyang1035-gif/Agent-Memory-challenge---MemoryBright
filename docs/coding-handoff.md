# 编码交接文件（Coding Handoff）

> 供测试 Agent / 后续维护者快速获取项目上下文，无需重复询问。
> 生成时间：2026-09-30；**最近更新：2026-10-19（同步当前最优配置与最新结论）**

## 1. 项目概览

- **目标**：实现符合 AML（Agent Memory Leaderboard）平台契约的"记忆服务"，在本地基准上超越 Mem0 / TencentDB 对照实现。
- **仓库根**：`/workspace/user_data/agent-memory/Agent-Memory-Challenge-server-20260929/Agent-Memory-Challenge`
- **自研服务目录**：`<仓库根>/memory-service`
- **Mem0 对照目录**：`<仓库根>/memory-service/mem0_service`
- **TencentDB 对照目录**：`<仓库根>/memory-service/tencentdb_service`
- **本地 embedding 服务**：`<仓库根>/memory-service/embed_server.py`（bge-m3，OpenAI 兼容，端口 8080）
- **比赛 harness（独立目录）**：`/workspace/user_data/agent-memory/Agent-Memory-Challenge-server-20260929/competition-starter-kit`（`AGENTS.md` 为唯一入口）

## 2. 运行方式（本地自测）

**推荐：一键启动 4 个服务**（在 `memory-service` 下执行）：

```bash
bash start_all.sh            # 重启全部（embed + self + mem0 + tencentdb）
bash start_all.sh status     # 只看状态（4 个 /health）
bash start_all.sh stop       # 只停止
```

- 使用 `setsid` 脱离会话，pid 文件在 `memory-service/.run/`，日志在 `/tmp/amc_logs/`。
- embed 服务首次预加载模型约 20~40s，脚本会轮询 health 后再拉起其余服务。

| 服务 | 端口 | 工作目录 | 启动命令 |
|---|---|---|---|
| 本地 embedding (bge-m3) | 8080 | `memory-service` | `python3 -u embed_server.py` |
| 自研记忆服务 | 8000 | `memory-service` | `python3 -u main.py` |
| Mem0 对照服务 | 8001 | `memory-service/mem0_service` | `python3 -u main.py` |
| TencentDB 对照服务 | 8002 | `memory-service/tencentdb_service` | `SERVICE_PORT=8002 python3 -u main.py` |

- 配置来源：`memory-service/.env`（模板见 `.env.example`）；**API Key 只允许从环境变量注入，禁止提交**。
- 依赖安装：`pip install -r memory-service/requirements.txt`（Mem0 见 `mem0_service/requirements.txt`）。
- 向量库：`QDRANT_HOST=local` 时使用**本地文件模式**，数据在 `memory-service/qdrant_storage`（**无需 Docker**）；元数据在 `memory-service/memory.db`（SQLite）。
- Linux 终端建议 `PYTHONIOENCODING=utf-8`；长命令建议放后台进程。

## 3. HTTP 接口（AML 契约）

### `GET /health`
- 无鉴权，返回 `{"status": "ok"}`（embed 服务额外返回 `model`/`dims`）。

### `POST /add`（同步写入，幂等）
```json
{
  "request_id": "conv-26-session_1-0",
  "user_id": "locomo_conv-26",
  "session_id": "session_1",
  "messages": [{"role": "user", "content": "...", "timestamp": 1688362560000}]
}
```
- 幂等：同一 `request_id` 重复写入直接返回 200；须原样回显 `request_id`/`user_id`/`session_id`。
- 内部流程：分块（注入日期前缀）→ embedding → LLM 分类 → 存 SQLite+Qdrant → 抽原子事实（`kind='fact'`）→ BM25 索引 → 冲突治理。

### `POST /search`（返回 `{"data":[...]}`，条数 ≤ top_k）
```json
{"query": "...", "user_id": "locomo_conv-26", "top_k": 10, "options": ["A","B",...]}
```
- 严格按 `user_id` 隔离；响应必须是 `{"data":[{id,content,score?,created_at?}]}`，不得超 `top_k`。
- 内部流程：query 理解（MCQ 走快路径）→ embedding（主 query + 选项向量合并成一次批量请求）→ 向量 + BM25 混合召回（各 200）→ 治理过滤 → 实体扩展（含分数衰减）→ Option-Overlap 加成 → 纯融合分排序（`RERANK_MODE=off`，不调 LLM 精排）→ Passage 回链（事实附加源分块完整原文）→ top_k。

## 4. 关键实现文件

| 文件 | 职责 |
|---|---|
| `main.py` | FastAPI 入口、Add/Search 编排、Option-Overlap、Passage 回链 |
| `config.py` | 全部配置项与环境变量解析 |
| `chunker.py` | 语义分块 + **会话日期前缀注入** |
| `fact_extractor.py` | 写入侧原子事实抽取（含日期解析、语言守卫） |
| `classifier.py` / `query_analyzer.py` | Add 侧分类 / Search 侧查询理解 |
| `reranker.py` | 融合打分 / LLM rerank（**当前 `RERANK_MODE=off`，走 fallback 融合分**） |
| `store.py` | Qdrant + SQLite 存取、schema 迁移、实体索引 |
| `bm25_index.py` / `governance.py` | BM25 增量索引 / 记忆治理（含 `GOVERNANCE_IMPORTANCE_BOOST`） |
| `time_normalizer.py` | 写入侧相对时间归一化（`ENABLE_TIME_NORMALIZE`；实测冗余，保留） |
| `embedder.py` / `llm.py` | 内部 embedding / LLM 客户端（含重试） |
| `embed_server.py` | 本地 bge-m3 embedding 服务（8080） |

### TencentDB 对照服务（`tencentdb_service/`）

| 文件 | 职责 |
|---|---|
| `tc_config.py` | 先载上级 `.env`，再用本目录 `.env` 覆盖（TCVDB_*、SERVICE_PORT=8002、BACKEND） |
| `tc_store.py` | 存储抽象：`tencent`（腾讯云向量数据库）/ `memory`（进程内）两后端，接口一致 |
| `main.py` | 8002 FastAPI，AML 契约；复用上级 chunker/embedder/reranker/fact_extractor |
| `smoke_test.py` / `README.md` / `.env.example` | 自测、说明、配置模板（凭据模板） |

- 当前 `BACKEND=memory`（进程内后端，仅流程验证）；启用真实腾讯云：把 `TCVDB_URL`/`TCVDB_KEY` 填入 `tencentdb_service/.env`，设 `BACKEND=tencent`，重启即可（embedding/LLM 继承上级 `.env`）。

## 5. 当前最优配置（已固化到 `.env`）

> 结论来源：`OPTIMIZATION.md` 第七节「优化点 1~6」+ 第八节「最终结论（2026-10-14）」。
> Opt2 / Opt5 / Opt6 均为**严格 Pareto 改进（零回退）**，已全部固化。

```
# —— 写入侧 ——
CHUNK_DATE_PREFIX=true            # 注入会话日期前缀
CHUNK_SPLIT_ON_SPEAKER=false      # 跨说话人聚合
ENABLE_FACT_EXTRACTION=true       # 原子事实抽取（双表示：fact + 原 chunk）
FACT_DEDUP_THRESHOLD=0.95
ENABLE_TIME_NORMALIZE=true        # Opt4：仅对事实做相对时间归一化（实测冗余，保留）

# —— 实体扩展（Opt6：修正候选分塌陷，关键修复）——
ENABLE_ENTITY_EXPANSION=true
ENTITY_EXPAND_TOP=5
ENTITY_EXPAND_LIMIT=30
ENTITY_EXPAND_SCORE_DECAY=0.5     # 0.9 -> 0.5（避免扩展候选与真实召回同分并列淹没 top-k）

# —— 检索侧（Opt2：减负，纯融合分）——
RERANK_MODE=off                   # 关闭 LLM 精排，走 fallback 融合分
ENABLE_CONDITIONAL_RERANK=false   # 关闭条件重排（aic 头号指标 + 延迟最低）
FALLBACK_VECTOR_WEIGHT=0.5
FALLBACK_BM25_WEIGHT=0.5
PREFER_FACTS=true
FACT_BOOST=0.10
DIVERSITY_PER_SESSION_CAP=1       # Opt1：每 session 最多 1 条候选
PASSAGE_MAX_CHARS=2000            # Passage 回链不截断（源分块完整原文）

# —— 候选池 ——
ENABLE_BM25=true
VECTOR_TOP_K=200
BM25_TOP_K=200

# —— 治理（Opt5：关闭 importance「富者愈富」反馈环）——
GOVERNANCE_IMPORTANCE_BOOST=0.0   # 检索返回后不再累加 importance，排序回到纯融合分
GOVERNANCE_IMPORTANCE_CAP=2.0

# —— 模型接入 ——
EMBEDDING_API_URL=http://localhost:8080/v1/embeddings   # 本地 bge-m3
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024
LLM_API_URL=https://ecloud.10086.cn/api/query/aigateway/v1/chat/completions  # ecloud 移动云网关
LLM_MODEL=zhanlu/glm-5.2                  # 小模型档（分类 / query 理解）
LLM_MODEL_LARGE=zhanlu/qwen3.7-max        # 大模型档（rerank / 治理）
LLM_TIER_CLASSIFY=small
LLM_TIER_QUERY_ANALYSIS=small
LLM_TIER_RERANK=large
LLM_TIER_GOVERNANCE=large
QDRANT_HOST=local                         # 本地文件模式，无需 Docker
```

> 注：`RERANK_CANDIDATE_LIMIT=15` 仅在 `RERANK_MODE!=off` 时生效；当前 off，不参与排序。

## 6. 评测

- Harness：`memory-service/bench/run_bench.py`（**禁止修改**，保证公平对比）。
- 数据集：`memory-service/bench/datasets/`（LoCoMo-MC10 / LongMemEval / CL-bench / PersonaMem 等）。
- 命令：
  ```
  python bench/run_bench.py --dataset locomo --sample 1 --per-conv 30 --max-questions 30 --top-k 10 --seed 42 --base-url http://127.0.0.1:8000
  ```

### 最终三方对比（LoCoMo 冻结库 conv-48，n=30，2026-10-14）

| 服务 | answer_in_context | keyword_ratio | top1_hit | 延迟 |
|---|---|---|---|---|
| **self（Opt2+Opt5+Opt6，8000）** | **46.7%** | **64.0%** | **60.0%** | 9.43s |
| mem0（8001） | 33.3% | 51.9% | 50.0% | 9.01s |
| tencentdb（8002，BACKEND=memory） | 40.0% | 53.9% | 43.3% | 13.68s |

> self 在 aic / kw / top1 三项**全面领先**；延迟与 mem0 相当、显著低于 tencentdb。
> **统计注意**：n=30 时每题占 3.3%，p≈0.45 时二项 SE≈9.1%，95% CI≈±18%——self 对 tencentdb 的 aic +6.7pt 处于临界；对 mem0 的 +13.4pt 较稳健。

### LongMemEval 冒烟（n=1，2026-10-14）

同一题 `lme_c7dc5443`（knowledge-update，46 sessions / 478 msgs）三方 aic/kw/top1 均 100%，**无区分度**；单题单服务 Add ≈ 28~39 min，故仅作"可跑通"证据，不追求统计意义。

### CL-bench 四方对比（sample=2 max-questions=2，2026-10-14）

| 服务 | keyword_ratio | top1_hit | rubric_coverage | avg_latency(s) |
|---|---|---|---|---|
| 自研 8000 | 46.7% | 100% | 53.9% | ~16.7 |
| mem0 8001 | 11.7% | 100% | 17.3% | 0.189 |
| 比赛版 tencentdb 8002 | 46.7% | 100% | 53.9% | ~28.8 |
| 真 TencentDB 8003 (MemoryCore) | 38.3% | 100% | 45.0% | 0.353 |

详见 `bench/COMPARISON.md`、`bench/FOUR_WAY_COMPARISON.md`、`bench/RESULTS.md`、`OPTIMIZATION.md`。

## 7. 应用信息

| 项 | 值 |
|---|---|
| 后端地址（本地） | `http://127.0.0.1:8000`（自研）/ `:8001`（mem0）/ `:8002`（tencentdb）/ `:8080`（embedding） |
| 公网地址 | **未部署**（AML 平台要求服务公网 HTTPS 可达，待后续） |
| 前端地址 | （不适用，本项目无前端） |
| 登录凭据 | 本服务当前 `SERVICE_API_KEY` **未设置**（对本机无鉴权）；公网部署时须设置并按 Token/Bearer/X-Api-Key 校验 |
| 内部 LLM / Embedding 地址 | 见 `.env`（凭据不落库、不输出） |

## 8. 已知风险 / 待办

- **公网部署缺失**：AML 正式评测要求服务公网可达（HTTPS），当前仅本地；需补域名/证书/反代。
- **内部依赖外部网关**：Embedding（本地 bge-m3）稳定；LLM 走 ecloud 移动云网关，间歇性超时/断连会影响延迟与偶发 500，服务已内置 3 次重试。
- **检索延迟**：当前约 9.4s（`RERANK_MODE=off` 后已较旧版 13.4s 下降），主因是 query 理解 + embedding 等调用；如需进一步压低可探索并发化。
- **剩余弱项（LoCoMo conv-48，n=30）**：
  - `q13` / `q59`：仅差 1 个 gold_kw 未达 `aic=True`（跨 session 召回广度，已改善至 mem0 水平）。
  - 其余 13 题**不可赢**：7 题 adversarial（gold_kw 字面为 `answerable`，与内容无关）、6 题词形不匹配（presented≠presenting）等指标伪影。
- **CL-bench 已到数据上限**；LongMemEval 全量成本过高（数十小时）。
