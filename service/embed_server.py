"""bge-m3 本地 OpenAI 兼容 embedding 服务（NPU 优先，CPU 兜底）。

只暴露 /v1/embeddings 与 /health，供三个对照服务（自研 / mem0 / tencentdb）共用。
模型从本地磁盘加载，避免外部依赖。

设备选择（2026-11-01）：
  本机是 Ascend 910（2 chip × 64GB HBM），vLLM 用掉 85% 后每 chip 仍余约 12GB，
  bge-m3（~1.1GB fp16）完全可以放上去。实测 62 条文本：NPU 0.29s / CPU 空载 6.04s
  / CPU 有负载 38s，差 20~130 倍。因此默认 EMBED_DEVICE=npu:0；NPU 不可用时
  自动回退 CPU（此时按 TORCH_THREADS 限线程）。

并发模型（2026-11-01 修正）：
  同一进程内开多个模型副本 + 多个 OpenMP 线程组会互相抢 CPU 与内存带宽，
  6 worker × 6 线程实测比单进程 16 线程慢 20 倍（塌陷式变慢，非线性加速）。
  因此固定「单 worker」：NPU 上单实例已足够，CPU 上单实例也最快。
"""
import asyncio
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import List, Union

sys.stdout.reconfigure(encoding="utf-8")

MODEL_PATH = os.environ.get("BGE_M3_PATH", "/workspace/user_data/agent-memory/models/bge-m3")
PORT = int(os.environ.get("EMBED_SERVER_PORT", "8080"))
# 优先用 NPU；不可用时自动回退 CPU
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "npu:0")
# 容器配额 40 核；TORCH_THREADS × WORKERS 应 <= 40，留出余量给主服务与网关 I/O
TORCH_THREADS = int(os.environ.get("EMBED_TORCH_THREADS", "16"))
# 并发 encode 的工作线程数（每个线程一个独立模型实例，约 1~2.3GB 内存/显存）。
# 2026-11-01 实测多实例互相拖慢，固定单实例。
WORKERS = max(1, int(os.environ.get("EMBED_WORKERS", "1")))
# 单条输入的最大 token 数：bge-m3 默认 8192，长文本编码开销随 token 线性增长。
# 记忆分块的语义主要集中在前半段，截断到 1024 兼顾效果与速度。
MAX_SEQ_LEN = int(os.environ.get("EMBED_MAX_SEQ_LEN", "1024"))
# 批内单次 forward 的样本数（显存无关，CPU 上 32~64 效率都不错）
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
# 合并窗口：把这段时间内到达的请求攒成一批（越小延迟越低，越大吞吐越高）
BATCH_WINDOW = float(os.environ.get("EMBED_BATCH_WINDOW", "0.03"))
BATCH_MAX_TEXTS = int(os.environ.get("EMBED_BATCH_MAX_TEXTS", "64"))
BATCH_MAX_CHARS = int(os.environ.get("EMBED_BATCH_MAX_CHARS", "24000"))
# 文本 -> 向量 的 LRU 缓存：官方 smoke 会反复喂同一批记忆/问题，命中后完全跳过编码
CACHE_SIZE = int(os.environ.get("EMBED_CACHE_SIZE", "20000"))

# 本机 640 核：BLAS/OpenMP 默认按核数开线程池，叠加上容器 cgroup pids 上限(10000)
# 会在并发时触发 "libgomp: Thread creation failed" 使进程崩溃。此处兜底限流。
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = str(TORCH_THREADS)
# 等待策略设为被动，避免空闲线程自旋空转抢占 CPU
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from fastapi import FastAPI
from pydantic import BaseModel

_models: List = []               # 每个 worker 一个独立模型实例，避免共享权重并发 forward
_model_lock = threading.Lock()   # 模型加载串行化

_cache: "OrderedDict[str, List[float]]" = OrderedDict()
_cache_lock = threading.Lock()

_job_queue: "queue.Queue[_Job]" = queue.Queue()
_workers_started = False
_worker_lock = threading.Lock()


class EmbeddingRequest(BaseModel):
    model: str = "bge-m3"
    input: Union[str, List[str]]
    dimensions: int | None = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


class _Job:
    """一次 /v1/embeddings 请求，由工作线程填充 vectors/error 后回调事件循环。"""

    __slots__ = ("texts", "loop", "future", "started")

    def __init__(self, texts: List[str], loop, future):
        self.texts = texts
        self.loop = loop
        self.future = future
        self.started = time.time()


def _cache_get(text: str):
    with _cache_lock:
        vec = _cache.get(text)
        if vec is not None:
            _cache.move_to_end(text)
        return vec


def _cache_put(text: str, vec: List[float]) -> None:
    with _cache_lock:
        _cache[text] = vec
        _cache.move_to_end(text)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)


def _load_model(worker_id: int = 0):
    """加载（或复用）第 worker_id 个 bge-m3 实例（双重检查锁）。"""
    if worker_id < len(_models) and _models[worker_id] is not None:
        return _models[worker_id]
    with _model_lock:
        while len(_models) <= worker_id:
            _models.append(None)
        if _models[worker_id] is not None:
            return _models[worker_id]
        import torch

        # 设备解析：npu 需先 import torch_npu；任何一步失败都回退 CPU，
        # 保证 embedding 服务不会因为设备问题起不来（历史事故：8080 挂掉后
        # /add 全部降级为 lexical-only，检索退化为 BM25）。
        device = EMBED_DEVICE
        if device.startswith("npu"):
            try:
                import torch_npu  # noqa: F401
                if not torch.npu.is_available():
                    raise RuntimeError("torch.npu.is_available() is False")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[embed-server] NPU unavailable ({exc!r}), fallback to CPU",
                    flush=True,
                )
                device = "cpu"
        if device == "cpu":
            # CPU 编码时按配额设置线程数（进程级）
            try:
                torch.set_num_threads(TORCH_THREADS)
            except Exception:
                pass

        from sentence_transformers import SentenceTransformer

        # bge-m3 在 HF 上是 sentence-transformers 兼容的
        print(
            f"[embed-server] loading bge-m3 (worker {worker_id}) from {MODEL_PATH} "
            f"on {device}",
            flush=True,
        )
        t0 = time.time()
        model = SentenceTransformer(MODEL_PATH, device=device)
        try:
            # NPU/GPU 上 fp16 更快且显存减半；CPU 保持 fp32
            if device != "cpu":
                model = model.half()
        except Exception as exc:  # noqa: BLE001
            print(f"[embed-server] half() failed, keep fp32: {exc}", flush=True)
        try:
            model.max_seq_length = MAX_SEQ_LEN
        except Exception as exc:  # noqa: BLE001
            print(f"[embed-server] set max_seq_length failed: {exc}", flush=True)
        print(
            f"[embed-server] worker {worker_id} loaded in {time.time()-t0:.1f}s "
            f"(device={device}, workers={WORKERS}, max_seq_len={MAX_SEQ_LEN}, "
            f"batch_size={BATCH_SIZE}, window={BATCH_WINDOW}s)",
            flush=True,
        )
        _models[worker_id] = model
        return model


def _resolve(future, vectors, error) -> None:
    if future.done():
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(vectors)


def _worker_loop(worker_id: int = 0) -> None:
    """encode 工作线程：攒批 -> 一次 forward -> 按请求切分结果。"""
    try:
        model = _load_model(worker_id)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[embed-server] model load failed (worker {worker_id}): {exc!r}",
            flush=True,
        )
        return

    while True:
        first = _job_queue.get()
        jobs = [first]
        n_texts = len(first.texts)
        n_chars = sum(len(t) for t in first.texts)
        deadline = time.time() + BATCH_WINDOW
        while n_texts < BATCH_MAX_TEXTS and n_chars < BATCH_MAX_CHARS:
            remain = deadline - time.time()
            if remain <= 0:
                break
            try:
                nxt = _job_queue.get(timeout=remain)
            except queue.Empty:
                break
            jobs.append(nxt)
            n_texts += len(nxt.texts)
            n_chars += sum(len(t) for t in nxt.texts)

        texts: List[str] = []
        for job in jobs:
            texts.extend(job.texts)

        t0 = time.time()
        wait = t0 - min(j.started for j in jobs)
        try:
            embs = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=BATCH_SIZE,
            ).tolist()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[embed-server] encode failed n={len(texts)}: {exc!r}", flush=True
            )
            for job in jobs:
                job.loop.call_soon_threadsafe(_resolve, job.future, None, exc)
            continue
        dt = time.time() - t0

        offset = 0
        for job in jobs:
            k = len(job.texts)
            part = embs[offset:offset + k]
            offset += k
            for text, vec in zip(job.texts, part):
                _cache_put(text, vec)
            job.loop.call_soon_threadsafe(_resolve, job.future, part, None)

        if n_texts > 1 or dt > 0.5:
            print(
                f"[embed-server] batch w={worker_id} jobs={len(jobs)} texts={n_texts} "
                f"chars={n_chars} wait={wait:.2f}s encode={dt:.2f}s",
                flush=True,
            )


def _ensure_workers() -> None:
    global _workers_started
    if _workers_started:
        return
    with _worker_lock:
        if _workers_started:
            return
        started = 0
        for i in range(WORKERS):
            try:
                # 在主线程串行加载，避免多个 worker 同时抢锁/抢内存
                _load_model(i)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[embed-server] model load failed (worker {i}): {exc!r}",
                    flush=True,
                )
                continue
            threading.Thread(
                target=_worker_loop, args=(i,), name=f"embed-worker-{i}", daemon=True
            ).start()
            started += 1
        if started:
            _workers_started = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预加载模型与工作线程，避免首个请求触发加载而超时。"""
    _ensure_workers()
    print(
        f"[embed-server] {WORKERS} worker(s) preloaded, ready to serve",
        flush=True,
    )
    yield


app = FastAPI(title="bge-m3-embed-server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "bge-m3", "dims": 1024}


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = list(req.input)

    total_chars = sum(len(t) for t in texts)

    # 1) 缓存全命中：完全跳过编码（重复评测数据的高频路径）
    results: List[List[float] | None] = [None] * len(texts)
    missing: List[str] = []
    missing_pos: List[int] = []
    for i, text in enumerate(texts):
        vec = _cache_get(text)
        if vec is None:
            missing.append(text)
            missing_pos.append(i)
        else:
            results[i] = vec

    if missing:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        _ensure_workers()
        _job_queue.put(_Job(missing, loop, future))
        vectors = await future
        for pos, vec in zip(missing_pos, vectors):
            results[pos] = vec

    data = [
        EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(results)
    ]
    return EmbeddingResponse(
        data=data,
        model=req.model or "bge-m3",
        usage={
            "prompt_tokens": max(1, total_chars // 4),
            "total_tokens": max(1, total_chars // 4),
        },
    )


if __name__ == "__main__":
    import uvicorn

    print(
        f"[embed-server] starting on :{PORT}, model path {MODEL_PATH}",
        flush=True,
    )
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
