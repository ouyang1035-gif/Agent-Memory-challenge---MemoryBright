"""配置模块：从环境变量读取所有配置。

使用 python-dotenv 加载 .env 文件（如果存在）。
API Key 等敏感信息只允许通过环境变量注入，禁止硬编码。
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


# ---------- Embedding API ----------
EMBEDDING_API_URL: str = _get(
    "EMBEDDING_API_URL", "http://10.12.3.93:19997/v1/embeddings"
)
EMBEDDING_API_KEY: str | None = _get("EMBEDDING_API_KEY")
EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM: int = int(_get("EMBEDDING_DIM", "1024"))

# ---------- LLM API ----------
# 默认档位（小模型，快）：分类、query 理解等高频简单任务
LLM_API_URL: str = _get("LLM_API_URL", "http://10.12.3.31:9997/v1/chat/completions")
LLM_API_KEY: str | None = _get("LLM_API_KEY")
LLM_MODEL: str = _get("LLM_MODEL", "qwen-internal")

# 大模型档位（可选，质量优先）：rerank、冲突治理等语义判断任务
# 未配置 LLM_MODEL_LARGE 时，所有请求自动回退到默认档位。
LLM_MODEL_LARGE: str | None = _get("LLM_MODEL_LARGE")
LLM_API_KEY_LARGE: str | None = _get("LLM_API_KEY_LARGE")
LLM_API_URL_LARGE: str = _get("LLM_API_URL_LARGE", LLM_API_URL)

# 单次 LLM 请求超时（秒）
LLM_TIMEOUT_SECONDS: float = float(_get("LLM_TIMEOUT_SECONDS", "60"))

# 是否允许推理模型输出思考过程（reasoning）。默认关闭：
# glm-5.2 等推理模型会把 max_tokens 全部耗在 reasoning 上导致 content 为空，
# 关闭后直接产出 JSON。设为 true 可恢复思考模式。
LLM_ENABLE_THINKING: bool = _get("LLM_ENABLE_THINKING", "false").lower() in (
    "1", "true", "yes", "on"
)

# 关闭思考的传参方式：
#   top      —— 顶层 enable_thinking 字段（移动云网关认这个，vLLM 会静默忽略）
#   template —— chat_template_kwargs={"enable_thinking": false}（vLLM 认这个）
LLM_DISABLE_THINKING_STYLE: str = (
    _get("LLM_DISABLE_THINKING_STYLE", "top") or "top"
).lower()

# 单次 LLM 调用的最大尝试次数（含首次）。
# 2026-10-29：网关抖动时同一模型会随机返回 400「当前模型不健康」，
# 原来 1 次尝试 = 直接丢弃结果，官方 smoke 里记忆产量掉了 90%。
LLM_MAX_ATTEMPTS: int = int(_get("LLM_MAX_ATTEMPTS", "4"))

# 单次调用所有重试加起来的时间预算（秒）：超过即放弃，避免 /add 被拖到分钟级。
LLM_RETRY_BUDGET_SECONDS: float = float(_get("LLM_RETRY_BUDGET_SECONDS", "25"))

# 熔断：连续多少次网络级/网关瞬时失败后打开；打开后冷却多少秒再半开探测。
LLM_BREAKER_FAIL_THRESHOLD: int = int(_get("LLM_BREAKER_FAIL_THRESHOLD", "8"))
LLM_BREAKER_COOLDOWN_SECONDS: float = float(
    _get("LLM_BREAKER_COOLDOWN_SECONDS", "60")
)

# 同时在途的网关请求上限。网关侧并发一高就返回 502/429，
# 把并发压在阈值内可把「排队 → 502 → 重试 → 更堵」的雪崩打断。
LLM_MAX_CONCURRENCY: int = int(_get("LLM_MAX_CONCURRENCY", "3"))

# ---------- 按任务指定模型档位 ----------
# 取值：small（默认档）| large（大模型档；未配置则自动回退 small）
LLM_TIER_CLASSIFY: str = _get("LLM_TIER_CLASSIFY", "small")
LLM_TIER_QUERY_ANALYSIS: str = _get("LLM_TIER_QUERY_ANALYSIS", "small")
LLM_TIER_RERANK: str = _get("LLM_TIER_RERANK", "large")
LLM_TIER_GOVERNANCE: str = _get("LLM_TIER_GOVERNANCE", "large")


def resolve_llm(tier: str = "small") -> tuple[str, str, str | None]:
    """把档位解析为 (model, api_url, api_key)。

    tier == "large" 且已配置 LLM_MODEL_LARGE 时用大模型，否则回退默认档位。
    """
    if tier == "large" and LLM_MODEL_LARGE:
        return LLM_MODEL_LARGE, LLM_API_URL_LARGE, LLM_API_KEY_LARGE
    return LLM_MODEL, LLM_API_URL, LLM_API_KEY

# ---------- Qdrant ----------
# 设为 "local" 使用本地文件模式（无需 Docker，数据存到 QDRANT_LOCAL_PATH）；
# 设为主机名/IP 则连接远程或 Docker 启动的 Qdrant 服务。
QDRANT_HOST: str = _get("QDRANT_HOST", "local")
QDRANT_PORT: int = int(_get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = _get("QDRANT_COLLECTION", "memories")
QDRANT_LOCAL_PATH: str = _get("QDRANT_LOCAL_PATH", "./qdrant_storage")

# ---------- SQLite ----------
SQLITE_DB_PATH: str = _get("SQLITE_DB_PATH", "./memory.db")

# ---------- 服务自身 ----------
SERVICE_HOST: str = _get("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT: int = int(_get("SERVICE_PORT", "8000"))
# 可选：如果配置了 SERVICE_API_KEY，则 /add 和 /search 需要 X-Api-Key 校验
SERVICE_API_KEY: str | None = _get("SERVICE_API_KEY")

# ---------- 行为参数 ----------
EMBEDDING_TIMEOUT_SECONDS: float = float(_get("EMBEDDING_TIMEOUT_SECONDS", "60"))
EMBEDDING_MAX_RETRIES: int = int(_get("EMBEDDING_MAX_RETRIES", "2"))
MAX_MESSAGES_PER_ADD: int = int(_get("MAX_MESSAGES_PER_ADD", "20"))

# ---------- 链路 2~5 功能开关 ----------
ENABLE_CHUNKING: bool = _get("ENABLE_CHUNKING", "true").lower() in ("true", "1", "yes")
ENABLE_BM25: bool = _get("ENABLE_BM25", "true").lower() in ("true", "1", "yes")
# 内存向量缓存：本地 Qdrant 的 payload 过滤是逐点扫描（无索引），collection 越大越慢；
# 把全量向量常驻内存后用 numpy 按 user 取子集算 cosine，单次检索 ~1ms。
ENABLE_VECTOR_CACHE: bool = _get("ENABLE_VECTOR_CACHE", "true").lower() in ("true", "1", "yes")
ENABLE_RERANK: bool = _get("ENABLE_RERANK", "true").lower() in ("true", "1", "yes")
ENABLE_CLASSIFICATION: bool = _get("ENABLE_CLASSIFICATION", "true").lower() in ("true", "1", "yes")
ENABLE_QUERY_ANALYSIS: bool = _get("ENABLE_QUERY_ANALYSIS", "true").lower() in ("true", "1", "yes")
ENABLE_GOVERNANCE: bool = _get("ENABLE_GOVERNANCE", "true").lower() in ("true", "1", "yes")

# ---------- 链路 2~5 调优参数 ----------
# 语义分块
CHUNK_TARGET_MIN: int = int(_get("CHUNK_TARGET_MIN", "150"))
CHUNK_TARGET_MAX: int = int(_get("CHUNK_TARGET_MAX", "250"))
CHUNK_TIME_GAP_MS: int = int(_get("CHUNK_TIME_GAP_MS", "3600000"))  # 1小时
# 是否按说话人切换切分。默认 false：跨说话人聚合到接近 CHUNK_TARGET_MAX 才切，
# 聚合时把每条消息格式化为 "{role}: {content}" 以保留说话人信息。
# true = 恢复旧行为（user/assistant 交替即切分）。
CHUNK_SPLIT_ON_SPEAKER: bool = _get("CHUNK_SPLIT_ON_SPEAKER", "false").lower() in (
    "true",
    "1",
    "yes",
)
# 是否在分块内容最前面注入会话日期前缀（形如 "[2023-07-03 July 2023] "），
# 便于把 "this month" 等相对时间与其真实日期关联，提升时间类问题的检索召回。
CHUNK_DATE_PREFIX: bool = _get("CHUNK_DATE_PREFIX", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 是否开启写入侧确定性相对时间归一化（P4）：把 "last year"/"yesterday" 等
# 英文相对时间按会话日期解析为绝对时间，以括号注记追加在原文之后
# （如 "last year" -> "last year (2022)"），提升时间类问题的检索召回。
ENABLE_TIME_NORMALIZE: bool = _get("ENABLE_TIME_NORMALIZE", "true").lower() in (
    "true",
    "1",
    "yes",
)

# 混合召回
VECTOR_TOP_K: int = int(_get("VECTOR_TOP_K", "50"))
BM25_TOP_K: int = int(_get("BM25_TOP_K", "50"))

# LLM Rerank
RERANK_CANDIDATE_LIMIT: int = int(_get("RERANK_CANDIDATE_LIMIT", "15"))
# 精排深度：只有融合分最高的前 N 条送 LLM，其余候选保持召回顺序（0 表示全部送）。
# top_k=100 时召回池 120 条，全量精排的输出 token 约 2400/请求，官方 17 并发下
# 实测单请求 p50 60s 且大面积超时降级；只精排头部可把成本降到 1/4，尾部仍按
# 融合分返回，召回宽度不变。
RERANK_DEPTH: int = int(_get("RERANK_DEPTH", "0"))
# 精排分批大小：候选数超过该值时拆成多批并发调用 LLM，压缩墙钟延迟
# （实测单批 10 条约 2.2s、40 条约 5.0s，拆 4 批并发后约 2.2~3.5s）。
RERANK_BATCH_SIZE: int = int(_get("RERANK_BATCH_SIZE", "12"))
# 滑窗精排步长：>0 时，头部候选按「窗口=RERANK_BATCH_SIZE、步长=本值」串行滑窗，
# 上一窗的排序结果作为下一窗 prompt 的锚点，使跨窗分数落在同一量纲上。
# 0 表示关闭（退回按批独立打分的旧行为，批与批之间分数不可比）。
RERANK_WINDOW_STRIDE: int = int(_get("RERANK_WINDOW_STRIDE", "0"))
RERANK_TIMEOUT_SECONDS: float = float(_get("RERANK_TIMEOUT_SECONDS", "20"))
# 重排整体墙钟上限（秒）：超过后取消未完成批次，未打分候选退化为融合分排序。
RERANK_TOTAL_TIMEOUT_SECONDS: float = float(
    _get("RERANK_TOTAL_TIMEOUT_SECONDS", "12")
)
# 网关（devbridge errfowt2）upstream timeout 实测为 60s：单次 /search 超过 60s
# 平台一律收到 504，等于白跑。这里强制收敛为单窗口精排 + 45s 墙钟上限，
# 不受外部环境变量影响（外层守护进程会用旧 env 重启服务）。
RERANK_DEPTH = min(RERANK_DEPTH or RERANK_BATCH_SIZE, RERANK_BATCH_SIZE)
RERANK_WINDOW_STRIDE = 0
RERANK_TOTAL_TIMEOUT_SECONDS = min(RERANK_TOTAL_TIMEOUT_SECONDS, 45.0)
# /search 总时间预算（秒）：超出则放弃精排、按融合分返回；0 表示不限制
SEARCH_TOTAL_BUDGET_SECONDS: float = float(
    _get("SEARCH_TOTAL_BUDGET_SECONDS", "0")
)
# 召回/精排候选深度上限：实际深度 = max(VECTOR_TOP_K, min(top_k, 本值))。
# 官方请求固定 top_k=100；若召回深度与精排候选池都写死 50/40，
# 大数据量用户会出现「答案存在但从未进入候选」的召回损失。
RECALL_DEPTH_MAX: int = int(_get("RECALL_DEPTH_MAX", "120"))
# rerank prompt 中每条候选内容的最大字符数，超出则截断（0 表示不截断）。
RERANK_MAX_CHARS: int = int(_get("RERANK_MAX_CHARS", "320"))
RERANK_WEIGHT: float = float(_get("RERANK_WEIGHT", "0.7"))
VECTOR_WEIGHT: float = float(_get("VECTOR_WEIGHT", "0.3"))
# RERANK_MODE=off 时的纯融合分权重（向量分 vs BM25 分），运行时自动归一化
FALLBACK_VECTOR_WEIGHT: float = float(_get("FALLBACK_VECTOR_WEIGHT", "0.7"))
FALLBACK_BM25_WEIGHT: float = float(_get("FALLBACK_BM25_WEIGHT", "0.3"))

# 查询理解
QUERY_ANALYSIS_TIMEOUT_SECONDS: float = float(_get("QUERY_ANALYSIS_TIMEOUT_SECONDS", "20"))
QUERY_ANALYSIS_CACHE_SIZE: int = int(_get("QUERY_ANALYSIS_CACHE_SIZE", "512"))

# 记忆治理
GOVERNANCE_IMPORTANCE_BOOST: float = float(_get("GOVERNANCE_IMPORTANCE_BOOST", "0.05"))
GOVERNANCE_IMPORTANCE_CAP: float = float(_get("GOVERNANCE_IMPORTANCE_CAP", "2.0"))

# ---------- P1：写入侧原子事实抽取（双表示）----------
# 开启后，每个分块额外经 LLM 抽取原子事实，作为独立记忆单元（kind='fact'）入库，
# 原始分块仍照旧入库（kind='chunk'）。
ENABLE_FACT_EXTRACTION: bool = _get("ENABLE_FACT_EXTRACTION", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 写入时事实去重：与同 user 已有事实的余弦相似度 ≥ 该阈值则跳过
FACT_DEDUP_THRESHOLD: float = float(_get("FACT_DEDUP_THRESHOLD", "0.95"))
# 每个分块最多抽取多少条原子事实
FACT_MAX_PER_CHUNK: int = int(_get("FACT_MAX_PER_CHUNK", "8"))
# 事实抽取失败（网关抖动）后的补抽轮数与每轮间隔基数（秒）
FACT_RETRY_ROUNDS: int = int(_get("FACT_RETRY_ROUNDS", "2"))
FACT_RETRY_DELAY_SECONDS: float = float(_get("FACT_RETRY_DELAY_SECONDS", "2.0"))

# ---------- P2：实体关联索引 + 多跳扩展 ----------
# 开启后维护 entity -> memory_id 倒排索引，检索时沿命中记忆的实体做 1 跳扩展。
ENABLE_ENTITY_EXPANSION: bool = _get("ENABLE_ENTITY_EXPANSION", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 扩展时取命中候选集中出现频次最高的前 N 个实体
ENTITY_EXPAND_TOP: int = int(_get("ENTITY_EXPAND_TOP", "5"))
# 每次检索最多补充多少条关联记忆
ENTITY_EXPAND_LIMIT: int = int(_get("ENTITY_EXPAND_LIMIT", "20"))
# 扩展候选的合成分数 = 种子记忆强度 × 该衰减系数。
# 必须显著低于真实召回命中（否则同分并列会淹没 top-k），默认 0.9 为历史行为。
ENTITY_EXPAND_SCORE_DECAY: float = float(_get("ENTITY_EXPAND_SCORE_DECAY", "0.9"))

# ---------- P3：检索侧减负（可开关，便于 A/B）----------
# hard = 现有 0~10 整数 LLM 打分；off = 不调 LLM，按向量/BM25 融合分排序
RERANK_MODE: str = (_get("RERANK_MODE", "hard") or "hard").lower()
# 条件 rerank：RERANK_MODE=off 时，仅当头部候选融合分过于接近（难区分）才调 LLM 精排
ENABLE_CONDITIONAL_RERANK: bool = _get("ENABLE_CONDITIONAL_RERANK", "false").lower() in (
    "true",
    "1",
    "yes",
)
# 头部 top_k 候选中，最高分与最低分差值小于该阈值即视为"分差接近"，触发 LLM rerank
CONDITIONAL_RERANK_GAP: float = float(_get("CONDITIONAL_RERANK_GAP", "0.03"))
# 排序时给 kind='fact' 的记忆一个小加成，让干净的事实优先于长分块
PREFER_FACTS: bool = _get("PREFER_FACTS", "true").lower() in ("true", "1", "yes")
FACT_BOOST: float = float(_get("FACT_BOOST", "0.05"))

# ---------- v3 ①：RRF 倒数排名融合（替代手工加权） ----------
# 依据：Cormack et al. (SIGIR 2009) 的 RRF：score(d)=Σ 1/(k+rank)；
# 2026《Training-Free Lexical–Dense Fusion for Conversational-Memory Retrieval》
# 在会话记忆检索上复验 lexical+dense 的 rank 融合优于手工加权。
# v2 的 0.5*cosine + 0.5*bm25_norm 有两个问题：两路量纲不同；BM25 的 top1 被
# 归一到 1.0，导致"只撞了字面词的弱匹配"也能拿到接近向量的满分。
ENABLE_RRF: bool = _get("ENABLE_RRF", "true").lower() in ("true", "1", "yes")
# RRF 平滑常数，论文与工程实践的常用默认值
RRF_K: int = int(_get("RRF_K", "60"))
# 纯 RRF 只看排名、丢弃分数量级：单通道深位的长分块会被"双通道命中的短事实"整体挤出
# （实测：A1-job 的答案分块从 rank24 掉到 100 名外）。故不做凸组合替换，而是把 RRF
# 作为**加性重排信号**叠加在 v2 分数融合之上：final = base + RRF_BONUS * rrf_norm。
# base 的量级差（~0.1~0.6）大于 bonus，基础排序不被推翻，只让"两路都靠前"的候选上浮。
# 0 表示关闭（完全等价 v2）。
# 实测（本地 19 题，官方正式评测固定 top_k=100 且按 Answer 模型+判分器给分）：
#   0.10 → hit@3 0.737→0.789、hit@20 0.842→0.895、cover@10 0.375→0.417；
#   hit@1 0.579→0.526（唯一变化是 D2 由第 1 名变第 2 名，两条都是正确轨迹证据）。
# 在 k=100 的判分范式下"证据进前列/覆盖更全"比"唯一第一名"更重要，故默认 0.10。
RRF_BONUS: float = float(_get("RRF_BONUS", "0.10"))

# ---------- MCQ 选项词面重叠加成 ----------
# 聚合/多跳型问题的答案常分散在多个 session，且正确选项与源记忆共享字面词
# （地点/物品名等）。对"与任一选项文本词面重叠度高"的候选择序时加成，
# 把分散在其它 session 的证据推进 top-k，提升跨 session 覆盖（无 options 时不变）。
ENABLE_OPTION_OVERLAP: bool = _get("ENABLE_OPTION_OVERLAP", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 最大加成幅度（按重叠覆盖率 0~1 线性缩放后累加到 final_score）
OPTION_OVERLAP_BOOST: float = float(_get("OPTION_OVERLAP_BOOST", "0.15"))

# ---------- Passage Integration：检索返回时把事实的源分块原文一并带上 ----------
# HippoRAG 2 思路：命中 kind='fact' 时，附加其 source_id 指向的原始分块原文，
# 既保留精准事实、又补回抽取时丢失的日期等关键细节。关闭时行为与旧版完全一致。
ENABLE_PASSAGE_INTEGRATION: bool = _get("ENABLE_PASSAGE_INTEGRATION", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 附加的源分块原文最大字符数（超出则截断）
PASSAGE_MAX_CHARS: int = int(_get("PASSAGE_MAX_CHARS", "600"))
# 源分块本身已在返回结果中时不再重复附加（避免同一原文出现两次）
PASSAGE_SKIP_IF_IN_HITS: bool = _get("PASSAGE_SKIP_IF_IN_HITS", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 同一源分块只给排位最高的那条事实附加一次（避免 N 条事实各带一份原文）
PASSAGE_ONCE_PER_SOURCE: bool = _get("PASSAGE_ONCE_PER_SOURCE", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 单次检索所有 [原文] 附件的总字符预算（<=0 表示不限制）
PASSAGE_TOTAL_BUDGET: int = int(_get("PASSAGE_TOTAL_BUDGET", "8000"))

# ---------- 返回体量控制（降噪）----------
# 最终分数归一化：FACT_BOOST / OPTION_OVERLAP_BOOST 叠加后可能 >1，
# 归一到 <=1 且保持相对顺序，避免分数溢出与非单调。
ENABLE_SCORE_NORMALIZE: bool = _get("ENABLE_SCORE_NORMALIZE", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 普通检索单次返回的总字符预算（按分数序累计，<=0 表示不限制）
SEARCH_TOTAL_CHARS_BUDGET: int = int(_get("SEARCH_TOTAL_CHARS_BUDGET", "60000"))

# ---------- P3 弃答门控 ----------
# 当最佳候选的精排分（0~10）低于阈值时判定为「不可回答」，返回空列表而非硬凑 top_k。
# 仅在精排真实生效（存在 rerank_score）时判断；精排失败/关闭时一律放行（fail-open）。
ENABLE_ABSTAIN_GATE: bool = _get("ENABLE_ABSTAIN_GATE", "false").lower() in (
    "true",
    "1",
    "yes",
)
ABSTAIN_MIN_RERANK: float = float(_get("ABSTAIN_MIN_RERANK", "3.0"))
# 精排覆盖率门槛：命中重排分数的候选占比低于此值时不做弃答判定（fail-open）。
ABSTAIN_MIN_COVERAGE: float = float(_get("ABSTAIN_MIN_COVERAGE", "0.5"))
# 预算裁剪时至少保留的条数（保证短问题也有足够证据）
SEARCH_MIN_ITEMS: int = int(_get("SEARCH_MIN_ITEMS", "8"))

# ---------- 摘要 / 全局模式 ----------
# 长历史全局综合（"summary of everything" 等）与普通检索的取向不同：
# 需要的不是 top-k 相关性，而是全量覆盖。开启后此类请求直接返回该用户全部记忆
# （分块优先、按时间顺序），并跳过 [原文] 附件与多样性裁剪。
ENABLE_SUMMARY_MODE: bool = _get("ENABLE_SUMMARY_MODE", "true").lower() in (
    "true",
    "1",
    "yes",
)
# top_k >= 该阈值即视为全局请求（官方 summary 请求传 500）
SUMMARY_TOP_K_THRESHOLD: int = int(_get("SUMMARY_TOP_K_THRESHOLD", "300"))
# 全局模式单次返回的总字符预算
SUMMARY_CHARS_BUDGET: int = int(_get("SUMMARY_CHARS_BUDGET", "120000"))

# ---------- As-of 时间锚定（"as of <date>" 查询的状态语义） ----------
# 此类查询问的是「某日期当时的状态」，应以该日期及之前最近事件为准，
# 而不是全历史里字面最相似（往往是通用画像）的记忆。仅在 query 含日+年级
# 显式日期时启用；按正文事件时间与 as_of 的间隔做衰减加成。
ENABLE_ASOF_BOOST: bool = _get("ENABLE_ASOF_BOOST", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 同日事件的最大加成（叠加在融合分数上）
ASOF_BOOST: float = float(_get("ASOF_BOOST", "0.15"))
# 衰减尺度：间隔 days 天后加成降为 ASOF_BOOST / (1 + days / 该值)
ASOF_DECAY_DAYS: float = float(_get("ASOF_DECAY_DAYS", "1.0"))

# ---------- 时间类提问的词面覆盖加成（"什么时候/多久/间隔"类查询） ----------
# 这类查询的答案通常是一句带时间表达的具体陈述，但句式相似的通用画像事实
# （如 "User 是一名……的人。"）常因字面相近占据 top1，把真正的答案压到第 5~10 位。
# 做法：按 query 关键词在候选正文中的覆盖率加权（词权重取候选池内 IDF 近似，
# 出现越少的词权重越大），只加不罚，未命中关键词的候选分数不变。
ENABLE_TEMPORAL_COVERAGE_BOOST: bool = _get(
    "ENABLE_TEMPORAL_COVERAGE_BOOST", "true"
).lower() in ("true", "1", "yes")
# 覆盖率 1.0 时的最大加成（叠加在融合分数上）
TEMPORAL_COVERAGE_BOOST: float = float(_get("TEMPORAL_COVERAGE_BOOST", "0.25"))

# ---------- 当前状态类提问的时序加成（"现在 / 目前 / 最近一次"） ----------
# D1（新值覆盖/当前状态）失分点：同一属性有多条历史陈述时，检索侧对时间不敏感，
# 旧值（"晨僵十来分钟"）会压住新值（"晨僵二十多分钟"）。
# 做法：query 含当前状态线索时，按 created_at 在候选池内做 min-max 归一化，
# 最旧 +0，最新 +RECENCY_BOOST；缺失 created_at 的不加不罚。
ENABLE_RECENCY_BOOST: bool = _get("ENABLE_RECENCY_BOOST", "false").lower() in (
    "true",
    "1",
    "yes",
)
RECENCY_BOOST: float = float(_get("RECENCY_BOOST", "0.15"))

# ---------- v3 ②：中文/无年份的日期锚定（"1月20日前后"） ----------
# 依据：Zep/Graphiti 的双时间记忆模型——检索时需要把"事件发生时间"作为一等
# 过滤/排序信号，而不只是看写入时间。v2 的日期解析只认带年份的显式日期
# （"January 20, 2024" / "2024-01-20"），中文会话里最常见的「1月20日前后」
# 「2月2日那次」全部解析失败，导致 C1（日期定位）类问题只能靠字面相似。
# 做法：识别无年份的「X月Y日」，用候选自身的 created_at / 正文日期按
# (月, 日) 比对，落在 ±DATE_ANCHOR_TOL_DAYS 内则加成（只加不罚）。
# 实测结论：默认关闭。C1-feb02-province（rank 11）在开启后掉到 26，原因是
# "2月2日那次"这类指代会在后续多个 session 里被反复提及，按 (月,日) 加成会
# 把"事后回顾"的分块抬到真正的原始事件之上。保留实现以便后续做 session 级去重。
ENABLE_DATE_ANCHOR: bool = _get("ENABLE_DATE_ANCHOR", "false").lower() in (
    "true",
    "1",
    "yes",
)
# 命中锚定日期的最大加成（叠加在融合分数上）
DATE_ANCHOR_BOOST: float = float(_get("DATE_ANCHOR_BOOST", "0.25"))
# 未显式说"前后/左右"时的默认容差（"那次/当天"等精确表述则为 0 天）
DATE_ANCHOR_TOL_DAYS: int = int(_get("DATE_ANCHOR_TOL_DAYS", "3"))

# ---------- v3 ③：当前状态类提问的"软失效"（不硬删历史值） ----------
# 依据：Mem0 最新版把陈旧性处理为 soft re-rank（0.3×–1.5× 时间衰减）而非硬删；
# Zep/Graphiti 用 edge invalidation（invalid_at/expired_at）保留历史边。
# v2 的 resolve_conflicts_latest 对 needs_latest 查询直接删掉同实体组的旧记忆，
# 会抹掉"变化轨迹"类问题（D2：晨僵从一开始到后来如何变化）所需的历史值。
# 做法：同实体组内非最新的候选乘 CONFLICT_STALE_PENALTY（降权不删除）。
ENABLE_SOFT_CONFLICT: bool = _get("ENABLE_SOFT_CONFLICT", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 旧值相对最新值的保留比例（0.85 = 降权 15%，仍可被召回）
CONFLICT_STALE_PENALTY: float = float(_get("CONFLICT_STALE_PENALTY", "0.85"))

# ---------- v2 ①：确定性派生「相对时间 → 绝对日期」事实 ----------
# 写入侧用纯规则（无 LLM）把分块里的 "半个月前 / 昨天 / 3 days ago" 按分块
# 时间戳解析成绝对日期，作为独立 kind='fact' 入库，直击 C1（日期/间隔）失分。
ENABLE_FACT_DERIVE: bool = _get("ENABLE_FACT_DERIVE", "true").lower() in (
    "true",
    "1",
    "yes",
)
# 每个分块最多派生几条（按信息强度排序后截断，user 段优先）
FACT_DERIVE_MAX_PER_CHUNK: int = int(_get("FACT_DERIVE_MAX_PER_CHUNK", "4"))

# ---------- v2 ②：事实去重保留「数值/时间不同」的冲突变体 ----------
# 旧行为：余弦相似度 ≥ FACT_DEDUP_THRESHOLD(0.95) 即丢弃新事实。
# 但句式几乎相同、只是数值/日期不同的两条事实（"INR 2 点多" vs "INR 3.2"）
# 相似度同样会超过阈值，结果新值被静默丢弃 → D1（新值覆盖）/D2（矛盾消解）失分。
# 开启后：两条事实的数值签名不同则都保留（冲突不丢，交给检索侧按时间排序）。
FACT_DEDUP_KEEP_NUMERIC_VARIANTS: bool = _get(
    "FACT_DEDUP_KEEP_NUMERIC_VARIANTS", "true"
).lower() in ("true", "1", "yes")

# ---------- v2 ③：纯口水话事实降 importance ----------
# 极窄规则：≤24 字、无数字、整句就是"好的/谢谢/我同意/ok/sounds good"之类。
# 偏好/规则类事实（E1~E3 已满分）一律不碰；只降 importance，不删记忆。
ENABLE_FILLER_IMPORTANCE_DOWNWEIGHT: bool = _get(
    "ENABLE_FILLER_IMPORTANCE_DOWNWEIGHT", "true"
).lower() in ("true", "1", "yes")
FILLER_IMPORTANCE_DELTA: float = float(_get("FILLER_IMPORTANCE_DELTA", "-0.35"))

# ---------- v2 ④：摘要模式要求关键词（防止高 top_k 的 doc-QA 被误切） ----------
# 旧门控只看 top_k ≥ SUMMARY_TOP_K_THRESHOLD(300)，官方 "summary of everything"
# 用 top_k=500 命中，但若某个普通问答也用大 k 就会被误判成摘要。
# 开启后要求「大 k」且「query 含 summary/overview/everything/总结」双条件。
SUMMARY_REQUIRE_KEYWORD: bool = _get(
    "SUMMARY_REQUIRE_KEYWORD", "true"
).lower() in ("true", "1", "yes")

# ---------- v4 P0：摘要模式改按「意图」触发 ----------
# 官方 harness 固定 top_k=100，而 SUMMARY_TOP_K_THRESHOLD 默认 300，
# 因此旧门控（大 k + 关键词）在正式评测里永远不会命中，summary 分支等于死代码。
# 开启后只要 query 命中摘要关键词即进入摘要模式，不再要求大 top_k。
SUMMARY_GATE_BY_INTENT: bool = _get(
    "SUMMARY_GATE_BY_INTENT", "true"
).lower() in ("true", "1", "yes")

# ---------- v4 P0：摘要记忆层 ----------
# 为每个 session 生成一条 kind='summary' 的抽取式 digest（不调 LLM，零方差）。
# 摘要题（F1）直接命中这些 digest，而不是把上百条 chunk 硬塞进 top_k。
ENABLE_SUMMARY_LAYER: bool = _get(
    "ENABLE_SUMMARY_LAYER", "false"
).lower() in ("true", "1", "yes")
# session 内 chunk 少于该值时不出摘要（过短会话摘要价值低）
SUMMARY_SESSION_MIN_CHUNKS: int = int(_get("SUMMARY_SESSION_MIN_CHUNKS", "2"))
# 单条 digest 的字符上限
SUMMARY_MAX_CHARS: int = int(_get("SUMMARY_MAX_CHARS", "800"))

# ---------- v5 P0-c：召回保底（F1 触发的第二种形态） ----------
# 官方 F1 的 query 并不含「总结全部」措辞（实测为 "DB task index 289"、
# "ALF task index 86"、整段 CONVERSATION TRANSCRIPT、NTSB 摘录等），
# 意图门控零命中，摘要层在正式评测中从未生效。改为「召回保底」：
# 当常规检索几乎没有返回（n <= MAX_HITS）或最高分低于 MIN_TOP_SCORE 时，
# 用该用户的会话摘要 + 近期分块补齐到 FILL 条，保证这类请求至少拿到
# 覆盖整段历史的证据，而不是 0~2 条碎片。默认关闭。
ENABLE_RECALL_FLOOR: bool = _get(
    "ENABLE_RECALL_FLOOR", "false"
).lower() in ("true", "1", "yes")
# 触发条件一：常规检索返回条数不超过该值
RECALL_FLOOR_MAX_HITS: int = int(_get("RECALL_FLOOR_MAX_HITS", "2"))
# 触发前提：该用户在库里的记忆条数 >= 该值，否则「返回少」只是历史本身少，
# 不是召回失败（官方 smoke 里的探针用户只有 1~3 条记忆，补摘要纯属噪声）
RECALL_FLOOR_MIN_STORE_ROWS: int = int(_get("RECALL_FLOOR_MIN_STORE_ROWS", "30"))
# 触发条件二：常规检索最高分低于该值（有返回但都不可信）
RECALL_FLOOR_MIN_TOP_SCORE: float = float(
    _get("RECALL_FLOOR_MIN_TOP_SCORE", "0.35")
)
# 保底后的目标条数上限（不超过 top_k）
RECALL_FLOOR_FILL: int = int(_get("RECALL_FLOOR_FILL", "20"))

# ---------- v3 ⑤：全局综合题的时间轴覆盖选取 ----------
# F1（摘要/长历史全局综合）要的不是 top-k 相关性而是历史覆盖。官方 harness
# 固定 top_k=100，SUMMARY_TOP_K_THRESHOLD=300 的摘要模式永远不会触发，
# 因此改为按 query 关键词识别全局题，用「时间桶轮转」选取，让结果铺满历史。
ENABLE_GLOBAL_COVERAGE: bool = _get(
    "ENABLE_GLOBAL_COVERAGE", "true"
).lower() in ("true", "1", "yes")

# ---------- v3 ⑥：「变化轨迹」题跳过软失效降权 ----------
# D2 类问题（"从最初的 X 变化成当前多少"）需要旧值和新值同时在 top 位，
# 软失效的 stale 降权会把旧值压下去，故此类 query 跳过冲突降权。
ENABLE_TRAJECTORY_SKIP_CONFLICT: bool = _get(
    "ENABLE_TRAJECTORY_SKIP_CONFLICT", "true"
).lower() in ("true", "1", "yes")

# ---------- 错误 QA 反馈过滤（在线学习负样本） ----------
# 库中会写入 "Streaming online QA feedback" 记忆（含 predicted_answer 与 score）。
# score < 阈值 表示该预测答案是错的，其正文/派生事实会作为“看似正确的证据”
# 排到首位，诱导 Answer 重犯同一错误。检索时过滤掉错误反馈及其派生事实。
ENABLE_BAD_FEEDBACK_FILTER: bool = _get(
    "ENABLE_BAD_FEEDBACK_FILTER", "true"
).lower() in ("true", "1", "yes")
BAD_FEEDBACK_SCORE_THRESHOLD: float = float(
    _get("BAD_FEEDBACK_SCORE_THRESHOLD", "0.99")
)

# ---------- 检索结果多样性（去冗余） ----------
# 同一 session（created_at 相同）的近似分块会重复挤占 top-k，降低跨 session 证据覆盖。
# 最终选取时限制同一 created_at 返回的条数上限，超出的候选让位给其它 session 的候选。
# <=0 表示关闭该约束（保留原行为）。
DIVERSITY_PER_SESSION_CAP: int = int(_get("DIVERSITY_PER_SESSION_CAP", "2"))

# ---------- API I/O 日志（/add、/search 的入参/出参留痕） ----------
# 每次调用写两行 JSON（request / response）到 API_IO_LOG_DIR/api_io.jsonl，便于 grep/jq。
# 完整正文（含每条记忆 content）另写 api_io_full.jsonl，默认关闭以控制磁盘占用。
API_IO_LOG: bool = _get("API_IO_LOG", "true").lower() in ("true", "1", "yes")
API_IO_LOG_DIR: str = _get("API_IO_LOG_DIR", "/tmp/amc_logs")
API_IO_LOG_FULL: bool = _get("API_IO_LOG_FULL", "false").lower() in ("true", "1", "yes")
# 摘要日志轮转：单文件上限与保留份数
API_IO_LOG_MAX_BYTES: int = int(_get("API_IO_LOG_MAX_BYTES", str(64 * 1024 * 1024)))
API_IO_LOG_BACKUPS: int = int(_get("API_IO_LOG_BACKUPS", "5"))
# 完整正文日志轮转（单条响应可达数百 KB）
API_IO_FULL_MAX_BYTES: int = int(_get("API_IO_FULL_MAX_BYTES", str(512 * 1024 * 1024)))
API_IO_FULL_BACKUPS: int = int(_get("API_IO_FULL_BACKUPS", "3"))
# 摘要里单条 content 的预览字符数
API_IO_PREVIEW_CHARS: int = int(_get("API_IO_PREVIEW_CHARS", "160"))
