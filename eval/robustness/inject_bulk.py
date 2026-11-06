"""批量直灌干扰记忆（绕过 /add 的 chunker + 逐次 BM25 重建）。

目的：构建"记忆量放大"的高倍率实验臂（如 ×25）。写入路径与 /add 完全相同
（store.save_memories -> SQLite + Qdrant），只跳过 chunker（一条文本=一条记忆）
和分类 LLM（用默认元数据；实验服务已关闭 ENABLE_CLASSIFICATION）。
BM25 索引在首次检索时按 user 懒构建，因此无需在此重建。

用法（必须先设置好 SQLITE_DB_PATH / QDRANT_LOCAL_PATH，且该目录无服务占用）：
  SQLITE_DB_PATH=/tmp/rb_s3/memory.db QDRANT_LOCAL_PATH=/tmp/rb_s3/qdrant_storage \
  python3 inject_bulk.py --near 5000 --synth 15000 --seed 7
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

V5 = "/workspace/amc_eval/v5"
sys.path.insert(0, V5)
sys.path.insert(0, "/workspace/amc_eval/robustness")

import embedder  # noqa: E402
import store  # noqa: E402
import distractors  # noqa: E402

USER_ID = "u_56d3c8e3472cb8d469229c64f3234a2b714d0ca1dfe28d575e95e956305429c9"
BASE_TS = 1735689600000  # 2025-01-01
STEP_MS = 900000  # 15 分钟，避免被 chunker 语义影响（此处不经过 chunker，仅保证时间线自然）
DEFAULT_META = {
    "type": "event",
    "entities": [],
    "time_expr": None,
    "is_correction": False,
}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=int, default=0)
    ap.add_argument("--synth", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    rng = __import__("random").Random(args.seed)
    await embedder.init_client()
    store.init_store()
    texts = distractors.build_near(
        os.environ["SQLITE_DB_PATH"], args.near, rng
    ) + distractors.build_synth(args.synth, rng)
    rng.shuffle(texts)
    print(f"generated {len(texts)} texts", flush=True)

    t0 = time.time()
    done = 0
    for start in range(0, len(texts), args.batch):
        batch = texts[start:start + args.batch]
        vectors = await embedder.embed_texts(batch)
        msgs = [("user", t, BASE_TS + (start + i) * STEP_MS) for i, t in enumerate(batch)]
        store.save_memories(
            request_id=f"bulk-{args.seed}-{start // args.batch:06d}",
            user_id=USER_ID,
            session_id=f"bulk-{args.seed}",
            messages=msgs,
            vectors=vectors,
            metadata_list=[dict(DEFAULT_META) for _ in batch],
        )
        done += len(batch)
        if (start // args.batch + 1) % 20 == 0:
            print(f"  injected {done}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(texts)} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
