import asyncio
import os
import random
import sys
import time

os.environ["RERANK_DEPTH"] = "32"
os.environ["RERANK_MAX_CHARS"] = "220"
os.environ["RERANK_BATCH_SIZE"] = "8"
os.environ["RERANK_TIMEOUT_SECONDS"] = "30"
os.environ["RERANK_TOTAL_TIMEOUT_SECONDS"] = "40"
os.environ["LLM_API_URL"] = "http://127.0.0.1:8090/v1/chat/completions"
os.environ["LLM_API_KEY"] = "local-no-auth"
os.environ["LLM_MODEL"] = "qwen3.8-27b"
os.environ["LLM_DISABLE_THINKING_STYLE"] = "template"
os.environ["LLM_ENABLE_THINKING"] = "false"

sys.path.insert(0, "/workspace/amc_eval/v5")

import config  # noqa: E402
import reranker  # noqa: E402

PARA = "用户提到自己上个月在市三医院做了胃镜检查，结果是慢性浅表性胃炎伴幽门螺杆菌阳性，医生开了两种药并叮嘱两周后复查。"
PARA2 = "用户还说自己对青霉素过敏，之前服用后出现皮疹，之后开药需要主动告知医生。"


def make_cands(n):
    random.seed(0)
    cands = []
    for i in range(n):
        content = (PARA if i % 3 == 0 else PARA2) * random.randint(1, 3)
        cands.append(
            {
                "id": f"mem_{i:04d}",
                "content": content,
                "score": round(random.uniform(0.3, 0.95), 4),
                "bm25_score": round(random.uniform(0.0, 8.0), 3),
                "rrf_score": round(random.uniform(0.0, 0.05), 5),
            }
        )
    return cands


async def main():
    print("depth=", config.RERANK_DEPTH, "chars=", config.RERANK_MAX_CHARS, "batch=", config.RERANK_BATCH_SIZE)
    for n in (60, 120):
        cands = make_cands(n)
        t0 = time.time()
        out = await reranker.rerank("我上次胃镜检查结果是什么，医生让我什么时候复查？", cands, top_k=100, candidate_limit=120)
        wall = time.time() - t0
        n_scored = sum(1 for c in out if c.get("rerank_score") is not None)
        top1 = out[0]
        print(
            f"cands={n} -> returned={len(out)} reranked={n_scored} wall={wall:.2f}s "
            f"top1={top1['id']} rs={top1['rerank_score']} fs={top1['final_score']:.4f}"
        )
        print("  前5:", [(c["id"], c["rerank_score"], round(c["final_score"], 4)) for c in out[:5]])
        print("  第33-35:", [(c["id"], c["rerank_score"], round(c["final_score"], 4)) for c in out[32:35]])


asyncio.run(main())
