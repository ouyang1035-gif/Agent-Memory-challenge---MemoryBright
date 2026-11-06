# Agent Memory Service（AMC 2026）

面向 Agent Memory Challenge 2026 / AML 平台评测的记忆服务：把对话历史写成**可分层、可治理、可检索**的记忆，并在 `top_k=100` 的检索契约下返回证据。

- 服务形态：FastAPI，`POST /add`、`POST /search`、`GET /health`
- 存储：SQLite（元数据 / 幂等 / 治理状态）+ 向量库（bge-m3 1024 维 cosine，可选 Qdrant 或内置索引）
- 模型：embedding 走 OpenAI 兼容端点；query 理解与精排走 qwen（27B / 小模型双档）
- 最终成绩：**full 53.19**（9,602 题全量；textual 5,609 + streaming 3,993），写入侧 38,183 次 `add` 全部提交

## 目录结构

```
agent-memory/
├── service/                 # 记忆服务源码（v5，线上运行版本）
│   ├── main.py              # FastAPI 入口，/add、/search、/health
│   ├── config.py            # 全部参数与钳制逻辑（含 .env 覆盖）
│   ├── chunker.py           # 语义分块
│   ├── classifier.py        # LLM 预分类
│   ├── fact_extractor.py    # 原子事实抽取
│   ├── store.py             # SQLite + 向量库读写
│   ├── reranker.py          # LLM 精排（批量打分 / 空结果重问 / 降级）
│   ├── query_analyzer.py    # query 理解与 MCQ 选项处理
│   ├── governance.py        # 冲突治理与 superseded 过滤
│   ├── bm25_index.py        # BM25 索引（按用户重建，默认关闭）
│   ├── embed_server.py      # 本地 embedding 服务
│   ├── start_all.sh         # 一键启动（embedding + 服务）
│   ├── .env.example         # 配置模板（无密钥）
│   └── README.md            # 早期链路 1 说明（历史文档）
├── eval/                    # 评测脚本与数据集
│   ├── run_eval.py          # 主评测（读写闭环）
│   ├── run_write_eval.py    # 写入侧评测
│   ├── replay.py            # 走查 / 复现单题
│   ├── eval_set.json        # 本地评测集
│   ├── queries.jsonl        # 本地 query 集
│   ├── expanded/            # 扩样回归集（60 题 / 分会话 / 分题型）
│   └── robustness/          # 鲁棒性探针（改写、负样本、冷启动、写入扩展）
├── reports/
│   ├── amc-memory-technical-report.html   # 技术报告（主线交付物）
│   ├── report.md                          # 报告文字版
│   ├── eval_reports/                      # 各版本评测输出
│   └── assets/、_shared/                   # 报告静态资源
├── logs/                    # 关键运行日志（.gz 压缩归档）
├── scripts/                 # 探针 / 压测 / 离线 A/B / 重启脚本
├── docs/                    # MANIFEST、coding handoff 等过程文档
├── archive/                 # 迁移自 /workspace 根目录的原始副本（已在 .gitignore 中排除）
│   ├── amc_eval/            # 原评测工作目录（含 v5 源码与各版本 store，约 11 GB）
│   ├── amc-memory-report/   # 原报告目录（与 reports/ 内容一致）
│   └── root-misc/           # 原根目录脚本与 memory.db
└── .gitignore
```

> `TencentDB-Agent-Memory-feat-server_team/`、`Agent-Memory-Challenge-server-20260929/` 是参考实现与官方赛题仓库，体积较大（数百 MB ~ 1.3 GB），未纳入版本库，需要时单独 clone。

## 架构

**写入（Add）**

```
幂等去重 → 语义分块 → embedding → LLM 预分类
        → 分块入库（kind=chunk）→ 实体索引
        → 原子事实抽取（kind=fact）→ 事实去重入库
        → 冲突治理（新值覆盖旧值，标记 superseded）
```

**检索（Search）**

```
query 理解 → 向量召回 top-N（默认 400）→（可选）BM25 召回与融合
          → 去重合并 → 实体多跳扩展 → 路由过滤 → superseded 过滤
          → LLM 精排 top-20 → 加权/事实加成 → 返回 top_k=100
```

设计取舍：**写厚、读薄**。写入侧尽量结构化（分块、事实、实体、治理状态都落库），检索侧只做「召回 + 重排 + 裁剪」，不做生成式改写，因此返回的始终是库内原文。

## 快速开始

```bash
cd service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # 填入 EMBEDDING_API_KEY / LLM_API_KEY
python main.py            # 默认 8000 端口
```

自测：

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/add -H 'Content-Type: application/json' -d '{
  "request_id": "demo:1",
  "user_id": "demo-user",
  "session_id": "demo-session",
  "messages": [{"role": "user", "timestamp": 1704067200000, "content": "我喜欢手冲咖啡。"}]
}'
curl -X POST http://localhost:8000/search -H 'Content-Type: application/json' -d '{
  "query": "用户喜欢什么饮品？", "user_id": "demo-user", "top_k": 10
}'
```

运行评测（服务需已启动）：

```bash
python eval/run_eval.py --base http://127.0.0.1:8000
python eval/expanded/run_expanded.py --base http://127.0.0.1:8000 --tag local
```

## 结果与结论

| 维度 | 得分 | 说明 |
|---|---|---|
| 总分 | **53.19** | 9,602 题全量，复用写入 |
| A 显式事实召回 | 54.69 | 单点事实可用，结构化字段抽取弱（A3 38.15） |
| B 关系与多跳推理 | 47.43 | 单条链路可恢复，跨片段证据聚合弱（B1 42.40） |
| C 时间与事件序列 | 49.18 | 绝对时间可算，顺序与轨迹弱 |
| D 记忆治理 | **64.56** | 覆盖/冲突/适用性强，删除抑制缺失（D3 20.00） |
| E 个性化与关怀 | 43.28 | 偏好可回忆，背景关怀弱 |
| G 上下文学习与执行 | 49.17 | 知识推理合格，执行与模拟几乎为零（G4 15.86） |
| H 认识论安全与隐私 | **66.62** | 拒答强（H1 76.41），最小披露弱（H2 37.56） |
| I 流式记忆 | 54.38 | 内部迁移极强（I2 95.32），在线学习一般 |
| F1 摘要与长历史综合 | 16.39 | 结构化 query 缺少入口 |

一句话：分数分布和链路结构高度一致——**能直接定位到单条记忆的题型在 50~76 分，需要跨条聚合 / 重算 / 按步骤执行的题型在 16~44 分**。详细拆解见 `reports/amc-memory-technical-report.html`。

## 数据与密钥

以下内容**不入库**，已在 `.gitignore` 中排除：

- `.env`（含真实 API Key）、`credentials.csv`、`api.txt`、`*_token.txt`
- `models/`（约 64 GB 本地模型权重）
- `tencentdb-real-data/`、各类 `*_store/`、`*.db`（运行态库，最大约 9.7 GB）
- 第三方参考仓库与原始日志（`*.log`）

如需复现，请从 `.env.example` 生成自己的 `.env`，并准备 embedding / LLM 两个 OpenAI 兼容端点。

## 已知限制

1. **题型入口**：官方 query 含 `task index 289`、`TRANSCRIPT`、`excerpt` 等结构化形态，当前 query 理解只处理自然语言，F1 类题命中率极低。
2. **聚合层缺失**：链路返回的是证据片段，没有「跨片段合并 + 重算」的中间层，B1/B3/C2/C3 类题受限。
3. **治理动作不全**：只有 superseded 覆盖，没有删除 / 遗忘 / 抑制接口，D3 仅 20.00。
4. **隐私取向**：`top_k=100` + 完整正文对召回最优，但对最小披露是反向的（H2 37.56）。
5. **延迟成本**：精排 `RERANK_MAX_CHARS=1000` × 20 候选，单题稳定约 30 s，是吞吐上限。
