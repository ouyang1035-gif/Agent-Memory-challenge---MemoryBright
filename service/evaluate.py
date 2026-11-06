# -*- coding: utf-8 -*-
"""本地召回质量评估器。

对每个 eval case：
1. 先 Add 所有对话
2. 再对每个 question 做 Search
3. 检查召回结果中是否包含 must_have 关键词（证据命中）、是否把 must_not 排最前（噪声）

指标：
- evidence_recall: must_have 关键词被召回的比例
- top1_correct: 第一条结果是否含任一 must_have
- noise_in_top1: 第一条是否含 must_not 词
"""
import json
import sys
import time
import urllib.request

from eval_dataset import EVAL_CASES

BASE = "http://127.0.0.1:8000"


def call(method, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def norm(text):
    return text.replace(" ", "").lower()


def evaluate():
    per_dim = {}
    total_q = 0
    total_ev_recall = 0.0
    total_top1 = 0
    total_noise_top1 = 0

    for case in EVAL_CASES:
        uid = case["user_id"]
        dim = case["dimension"]

        # Add 所有对话
        for i, conv in enumerate(case["conversations"]):
            s, b = call("POST", "/add", {
                "request_id": f"{uid}-add-{i}",
                "messages": conv,
                "user_id": uid,
                "session_id": f"{uid}-sess-{i}",
            })
            if s != 200:
                print(f"[ERROR] add failed for {uid}: {s} {b}")
                continue

        # Search 每个 question
        for q in case["questions"]:
            total_q += 1
            payload = {"query": q["query"], "user_id": uid, "top_k": 10}
            if q.get("options"):
                payload["options"] = q["options"]
            s, b = call("POST", "/search", payload)
            data = b.get("data", [])
            contents = [it["content"] for it in data]
            joined = norm(" ".join(contents))
            top1 = norm(contents[0]) if contents else ""

            must_have = q.get("must_have", [])
            must_not_top1 = q.get("must_not_in_top1", [])

            hit = sum(1 for k in must_have if norm(k) in joined)
            ev_recall = hit / len(must_have) if must_have else 1.0
            top1_ok = any(norm(k) in top1 for k in must_have) if must_have else True
            noise_top1 = any(norm(k) in top1 for k in must_not_top1)

            total_ev_recall += ev_recall
            total_top1 += 1 if top1_ok else 0
            total_noise_top1 += 1 if noise_top1 else 0

            d = per_dim.setdefault(dim, {"q": 0, "recall": 0.0, "top1": 0})
            d["q"] += 1
            d["recall"] += ev_recall
            d["top1"] += 1 if top1_ok else 0

            status = "OK " if (ev_recall == 1.0 and top1_ok and not noise_top1) else "   "
            print(f"[{status}] {dim:16s} | {q['query'][:22]:24s} | R={ev_recall:.2f} top1={'Y' if top1_ok else 'N'} noise={'Y' if noise_top1 else 'N'} | recall {len(data)}")

    print("\n===== 总览 =====")
    print(f"总问题数:        {total_q}")
    print(f"平均证据召回率:  {total_ev_recall/total_q:.2%}")
    print(f"Top1 命中率:     {total_top1/total_q:.2%}")
    print(f"Top1 噪声率:     {total_noise_top1/total_q:.2%}")

    print("\n===== 分维度 =====")
    for dim, d in per_dim.items():
        print(f"{dim:18s} | n={d['q']:2d} | recall={d['recall']/d['q']:.2%} | top1={d['top1']/d['q']:.2%}")


if __name__ == "__main__":
    evaluate()
