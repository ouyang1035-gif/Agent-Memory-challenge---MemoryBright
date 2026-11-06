#!/usr/bin/env python3
"""直接压 vLLM 的 rerank 批量调用形态，找出 8090 的真实吞吐上限。"""
import asyncio
import json
import statistics
import sys
import time

import httpx

URL = "http://127.0.0.1:8090/v1/chat/completions"
MODEL = "qwen3.8-27b"
API_KEY = "local-no-auth"

SYS = """你是一个记忆检索相关性评估助手。给定一个查询和多条记忆，为每条记忆评估其与查询的相关性，输出 0~10 的整数分数。

打分标准（必须严格执行）：
- 10：该记忆直接、明确地包含查询所问的答案（含查询中的实体，且时间/条件与查询一致）。
- 7~9：高度相关，包含答案的关键信息，但缺少个别细节。
- 4~6：主题相关，但不直接回答查询。
- 1~3：仅共享个别词或宽泛主题。
- 0：完全无关。

特别要求：
1) 若查询包含具体时间/时间范围（如某年某月、某天、某季节），只有时间一致的记忆才能给高分；时间明显不符的记忆最高不超过 3 分。
2) 若查询询问具体的人/物/地点/数量，缺少该要素的记忆不得给高分（最多 4 分）。
3) 泛化/寒暄/情绪类记忆（如"X 有家人""X 感到开心""X 喜欢艺术"），除非查询明确在问这些，否则最多 3 分。
4) 不得因为记忆更长、包含更多字就给高分；只按"是否直接回答查询"评分。
5) 为每一条记忆都必须给出分数，不得遗漏。

输出要求：
- 严格输出 JSON 数组，格式如 [{"id": "mem_xxx", "score": 8}, ...]
- id 必须与输入中的记忆 id 完全一致
- 只输出 JSON，不要输出任何其他解释文字
"""

PARA = "用户提到自己上个月在市三医院做了胃镜检查，结果是慢性浅表性胃炎伴幽门螺杆菌阳性，医生开了两种药并叮嘱两周后复查。"
PARA2 = "用户还说自己对青霉素过敏，之前服用后出现皮疹，之后开药需要主动告知医生。"
PARA3 = "用户表示最近工作压力较大，睡眠不好，经常凌晨两点才睡，希望得到改善建议。"


def make_prompt(query, n, chars):
    lines = [f"查询：{query}", "", "记忆列表："]
    for i in range(n):
        body = (PARA + PARA2 + PARA3) * 3
        lines.append(f"[mem_{i:04d}] {body[:chars]}")
    lines.append("")
    lines.append("请输出 JSON 数组，为每条记忆打分：")
    return "\n".join(lines)


async def one(client, prompt, max_tokens):
    t = time.time()
    try:
        r = await client.post(
            URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        body = r.json()
        txt = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return time.time() - t, len(txt), usage, None
    except Exception as exc:  # noqa
        return time.time() - t, 0, {}, f"{type(exc).__name__}: {exc}"


async def scenario(n_calls, batch, chars, max_tokens):
    prompt = make_prompt("我上次胃镜检查结果是什么，医生让我什么时候复查？", batch, chars)
    async with httpx.AsyncClient(timeout=300.0) as client:
        t0 = time.time()
        res = await asyncio.gather(*[one(client, prompt, max_tokens) for _ in range(n_calls)])
        wall = time.time() - t0
    lat = sorted(r[0] for r in res)
    errs = [r[3] for r in res if r[3]]
    tok = [r[2].get("prompt_tokens", 0) for r in res if r[2]]
    out = [r[2].get("completion_tokens", 0) for r in res if r[2]]
    p = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]
    print(
        f"calls={n_calls:4d} batch={batch:2d} chars={chars:3d} | wall={wall:6.2f}s "
        f"p50={p(0.5):6.2f}s p90={p(0.9):6.2f}s max={lat[-1]:6.2f}s "
        f"err={len(errs):2d} in_tok={statistics.mean(tok) if tok else 0:.0f} out_tok={statistics.mean(out) if out else 0:.0f}"
    )
    if errs:
        print("   err sample:", errs[0][:120])
    return wall


async def main():
    print("== A: batch16 x 300chars (max_tokens=1280) ==")
    for n in (1, 8, 24, 64, 136):
        await scenario(n, 16, 300, 1280)
        await asyncio.sleep(3)
    print("== B: batch8 x 300chars (max_tokens=768) ==")
    for n in (1, 8, 24, 64, 136):
        await scenario(n, 8, 300, 768)
        await asyncio.sleep(3)
    print("== C: batch8 x 200chars (max_tokens=768) ==")
    for n in (1, 24, 136):
        await scenario(n, 8, 200, 768)
        await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
