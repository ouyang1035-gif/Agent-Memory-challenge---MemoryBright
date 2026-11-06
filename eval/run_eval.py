#!/usr/bin/env python3
"""AMC v1 vs v2 本地回归评测器（只读，不改任何服务代码）。

用法:
  python3 run_eval.py                       # 跑 eval_set.json，双打 8020(v1)/8021(v2)
  python3 run_eval.py --set eval_set.json --out /tmp/amc_eval/report1
  python3 run_eval.py --replay /tmp/eval_dump/real_queries.jsonl   # 真实流量 A/B 稳定性

打分口径:
  hit@k   : 第 k 条之前出现「满足 gold 正则」的记忆
  MRR     : 1 / 首次命中排名
  cover   : 摘要题关键点覆盖（在所有返回里出现即算命中）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

TARGETS = {"v1": "http://127.0.0.1:8020", "v2": "http://127.0.0.1:8021"}


def post_search(base: str, user_id: str, query: str, top_k: int, timeout: float = 180.0) -> dict:
    payload = {
        "user_id": user_id,
        "query": query,
        "top_k": int(top_k),
        "request_id": f"eval-{int(time.time() * 1000)}-{abs(hash((base, query))) % 100000}",
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/search",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return {"resp": body, "elapsed": time.time() - t0}


def doc_matches(content: str, must: list[str], any_: list[str]) -> bool:
    for pat in must or []:
        if not re.search(pat, content, re.IGNORECASE):
            return False
    if any_:
        if not any(re.search(p, content, re.IGNORECASE) for p in any_):
            return False
    return True


def score_item(item: dict, resp: dict, elapsed: float) -> dict:
    docs = (resp.get("data") or []) if isinstance(resp, dict) else []
    contents = [d.get("content") or "" for d in docs]
    must = item.get("must") or []
    any_ = item.get("any") or []

    first_rank = 0
    if must or any_:
        for i, c in enumerate(contents, start=1):
            if doc_matches(c, must, any_):
                first_rank = i
                break

    out = {
        "id": item["id"],
        "cat": item.get("cat", "?"),
        "top_k": item["top_k"],
        "n_returned": len(contents),
        "elapsed": round(elapsed, 3),
        "first_rank": first_rank,
        "hit@1": 1 if 0 < first_rank <= 1 else 0,
        "hit@3": 1 if 0 < first_rank <= 3 else 0,
        "hit@5": 1 if 0 < first_rank <= 5 else 0,
        "hit@10": 1 if 0 < first_rank <= 10 else 0,
        "hit@20": 1 if 0 < first_rank <= 20 else 0,
        "hit@all": 1 if first_rank > 0 else 0,
        "mrr": round(1.0 / first_rank, 4) if first_rank else 0.0,
    }
    cov = item.get("coverage")
    if cov:
        joined = "\n".join(contents)
        hit_cov = sum(1 for p in cov if re.search(p, joined, re.IGNORECASE))
        top10 = "\n".join(contents[:10])
        hit_cov10 = sum(1 for p in cov if re.search(p, top10, re.IGNORECASE))
        out["coverage"] = round(hit_cov / len(cov), 4)
        out["coverage@10"] = round(hit_cov10 / len(cov), 4)
        out["coverage_detail"] = [p for p in cov if not re.search(p, joined, re.IGNORECASE)]
    return out


def agg(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    d = {
        "n": n,
        "hit@1": round(sum(r["hit@1"] for r in rows) / n, 3),
        "hit@3": round(sum(r["hit@3"] for r in rows) / n, 3),
        "hit@5": round(sum(r["hit@5"] for r in rows) / n, 3),
        "hit@10": round(sum(r["hit@10"] for r in rows) / n, 3),
        "hit@20": round(sum(r["hit@20"] for r in rows) / n, 3),
        "hit@all": round(sum(r["hit@all"] for r in rows) / n, 3),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 4),
        "avg_returned": round(sum(r["n_returned"] for r in rows) / n, 1),
        "avg_elapsed": round(sum(r["elapsed"] for r in rows) / n, 2),
    }
    cov = [r["coverage"] for r in rows if "coverage" in r]
    if cov:
        d["coverage"] = round(sum(cov) / len(cov), 3)
        cov10 = [r["coverage@10"] for r in rows if "coverage@10" in r]
        d["coverage@10"] = round(sum(cov10) / len(cov10), 3)
    return d


def run_items(args) -> int:
    with open(args.set, encoding="utf-8") as f:
        spec = json.load(f)
    users = spec["users"]
    items = spec["items"]
    if args.only:
        cats = set(args.only.split(","))
        items = [it for it in items if it.get("cat") in cats]

    results: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        uid = users[it["user"]] if it["user"] in users else it["user"]
        for name, base in TARGETS.items():
            try:
                r = post_search(base, uid, it["query"], it["top_k"], timeout=args.timeout)
                row = score_item(it, r["resp"], r["elapsed"])
            except Exception as exc:  # noqa: BLE001
                row = {"id": it["id"], "cat": it.get("cat", "?"), "top_k": it["top_k"],
                       "n_returned": 0, "elapsed": 0.0, "first_rank": 0,
                       "hit@1": 0, "hit@3": 0, "hit@5": 0, "hit@10": 0, "hit@20": 0,
                       "hit@all": 0, "mrr": 0.0, "error": f"{type(exc).__name__}: {exc}"}
            row["query"] = it["query"]
            results[name].append(row)

    print(f"\n{'=' * 96}\n本地评测集（{len(items)} 项）v1@8020 vs v2@8021\n{'=' * 96}")
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for name, rows in results.items():
        for r in rows:
            by_id[r["id"]][name] = r
    print(f"{'item':22} {'cat':4} {'k':>4} | {'v1 rank':>8} {'v2 rank':>8} | {'v1 ret':>6} {'v2 ret':>6} | "
          f"{'v1 cov':>6} {'v2 cov':>6} | {'v1 s':>5} {'v2 s':>5}")
    for it in items:
        d = by_id[it["id"]]
        a, b = d.get("v1", {}), d.get("v2", {})
        r1 = a.get("first_rank") or ("ERR" if "error" in a else "-")
        r2 = b.get("first_rank") or ("ERR" if "error" in b else "-")
        c1 = a.get("coverage", "-")
        c2 = b.get("coverage", "-")
        print(f"{it['id']:22} {it.get('cat','?'):4} {it['top_k']:>4} | {str(r1):>8} {str(r2):>8} | "
              f"{a.get('n_returned',0):>6} {b.get('n_returned',0):>6} | {str(c1):>6} {str(c2):>6} | "
              f"{a.get('elapsed',0):>5.2f} {b.get('elapsed',0):>5.2f}")

    print(f"\n{'-' * 96}\n按能力分组合计\n{'-' * 96}")
    cats = sorted({r["cat"] for rows in results.values() for r in rows})
    print(f"{'cat':5} {'n':>3} | {'v1 h@1':>7} {'v2 h@1':>7} | {'v1 h@10':>8} {'v2 h@10':>8} | "
          f"{'v1 mrr':>7} {'v2 mrr':>7} | {'v1 cov':>7} {'v2 cov':>7}")
    for cat in cats:
        aggs = {n: agg([r for r in rows if r["cat"] == cat]) for n, rows in results.items()}
        a, b = aggs["v1"], aggs["v2"]
        print(f"{cat:5} {a['n']:>3} | {a['hit@1']:>7.2f} {b['hit@1']:>7.2f} | {a['hit@10']:>8.2f} {b['hit@10']:>8.2f} | "
              f"{a['mrr']:>7.3f} {b['mrr']:>7.3f} | {a.get('coverage','-'):>7} {b.get('coverage','-'):>7}")

    print(f"\n{'-' * 96}\n总体\n{'-' * 96}")
    overall = {}
    for name, rows in results.items():
        overall[name] = agg(rows)
        o = overall[name]
        print(f"{name}: n={o['n']} hit@1={o['hit@1']} hit@3={o['hit@3']} hit@5={o['hit@5']} "
              f"hit@10={o['hit@10']} hit@20={o['hit@20']} hit@all={o['hit@all']} MRR={o['mrr']} "
              f"avg_ret={o['avg_returned']} avg_s={o['avg_elapsed']}"
              + (f" cover={o.get('coverage')} cover@10={o.get('coverage@10')}" if "coverage" in o else ""))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "items_result.json"), "w", encoding="utf-8") as f:
        json.dump({"targets": TARGETS, "overall": overall,
                   "detail": {n: rows for n, rows in results.items()}}, f, ensure_ascii=False, indent=2)
    print(f"\n明细已写入 {os.path.join(args.out, 'items_result.json')}")
    return 0


def run_replay(args) -> int:
    rows = [json.loads(line) for line in open(args.replay, encoding="utf-8") if line.strip()]
    print(f"真实 query 回放：{len(rows)} 条，v1@8020 vs v2@8021")
    stats = []
    errs = 0
    for i, q in enumerate(rows, 1):
        got = {}
        for name, base in TARGETS.items():
            try:
                r = post_search(base, q["user_id"], q["query"], q.get("top_k") or 10, timeout=args.timeout)
                docs = r["resp"].get("data") or []
                got[name] = {"elapsed": r["elapsed"], "docs": docs}
            except Exception as exc:  # noqa: BLE001
                errs += 1
                got[name] = {"elapsed": 0.0, "docs": [], "error": f"{type(exc).__name__}"}
        v1d, v2d = got["v1"]["docs"], got["v2"]["docs"]
        c1 = [d.get("content") or "" for d in v1d]
        c2 = [d.get("content") or "" for d in v2d]
        top1_same = bool(c1 and c2 and c1[0] == c2[0])
        top1_in_v2_5 = bool(c1 and c2 and c1[0] in c2[:5])
        top1_in_v1_5 = bool(c1 and c2 and c2[0] in c1[:5])
        s1, s2 = set(c1[:10]), set(c2[:10])
        jac = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 1.0
        stats.append({
            "user_id": q["user_id"], "query": q["query"], "top_k": q.get("top_k"),
            "n1": len(c1), "n2": len(c2),
            "t1": round(got["v1"]["elapsed"], 3), "t2": round(got["v2"]["elapsed"], 3),
            "top1_same": top1_same, "top1_v1_in_v2_5": top1_in_v2_5, "top1_v2_in_v1_5": top1_in_v1_5,
            "jaccard10": round(jac, 4),
            "v1_empty": not c1, "v2_empty": not c2,
        })
        if i % 50 == 0:
            print(f"  ... {i}/{len(rows)}")

    n = len(stats)
    def pct(key):
        return round(100.0 * sum(1 for s in stats if s[key]) / n, 2)
    summary = {
        "n": n, "errors": errs,
        "top1_same_pct": pct("top1_same"),
        "top1_v1_in_v2_5_pct": pct("top1_v1_in_v2_5"),
        "top1_v2_in_v1_5_pct": pct("top1_v2_in_v1_5"),
        "avg_jaccard10": round(statistics.mean(s["jaccard10"] for s in stats), 4),
        "v1_empty_pct": pct("v1_empty"), "v2_empty_pct": pct("v2_empty"),
        "avg_n1": round(statistics.mean(s["n1"] for s in stats), 2),
        "avg_n2": round(statistics.mean(s["n2"] for s in stats), 2),
        "avg_t1": round(statistics.mean(s["t1"] for s in stats), 3),
        "avg_t2": round(statistics.mean(s["t2"] for s in stats), 3),
        "p95_t1": round(sorted(s["t1"] for s in stats)[int(n * 0.95) - 1], 3),
        "p95_t2": round(sorted(s["t2"] for s in stats)[int(n * 0.95) - 1], 3),
    }
    print("\n" + "=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("=" * 80)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "replay_result.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detail": stats}, f, ensure_ascii=False, indent=2)
    print(f"明细已写入 {os.path.join(args.out, 'replay_result.json')}")

    diffs = [s for s in stats if not s["top1_same"]]
    print(f"\nTop1 不一致样例（前 12 条，共 {len(diffs)} 条）:")
    for s in diffs[:12]:
        print(f"  [{s['user_id'][:14]}] k={s['top_k']} jac={s['jaccard10']} | {s['query'][:70]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json"))
    ap.add_argument("--out", default="/tmp/amc_eval/report")
    ap.add_argument("--only", default="", help="只跑指定 cat，逗号分隔，如 A1,C1,F1")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--replay", default="", help="真实 query jsonl 回放模式")
    ap.add_argument("--base-v1", default="", help="覆盖 v1 地址，如 http://127.0.0.1:8020")
    ap.add_argument("--base-v2", default="", help="覆盖 v2 地址，如 http://127.0.0.1:8022")
    args = ap.parse_args()
    if args.base_v1:
        TARGETS["v1"] = args.base_v1.rstrip("/")
    if args.base_v2:
        TARGETS["v2"] = args.base_v2.rstrip("/")
    if args.replay:
        return run_replay(args)
    return run_items(args)


if __name__ == "__main__":
    sys.exit(main())
