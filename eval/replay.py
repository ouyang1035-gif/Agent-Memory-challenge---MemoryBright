#!/usr/bin/env python3
"""官方 smoke 调用重放：对本地 /search 复跑已记录的 query，检查召回是否退化。

数据来源：/tmp/amc_logs/api_io.jsonl（官方 /add、/search 留痕）
基线字段 base_n / base_chars / base_top_ids 为该 query 当时的返回。

用法：
  python3 replay.py                 # 全量重放
  python3 replay.py --round R4      # 只重放某一轮
  python3 replay.py --limit 5       # 只跑前 5 条（快速自检）
比较口径（只看检索侧，不判答案对错——官方 scorer 不可本地运行）：
  - n 条数是否明显变少（召回面收窄）
  - 返回总字符是否超预算（上下文污染）
  - 头部 10 条 id 是否漂移（排序稳定性）
"""
import argparse
import collections
import json
import urllib.request

SERVICE = "http://127.0.0.1:8000/search"
BASE = "/workspace/amc_eval/queries.jsonl"


def post(payload, timeout=300):
    req = urllib.request.Request(
        SERVICE,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/workspace/amc_eval/last_run.jsonl")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip()]
    if args.round:
        cases = [c for c in cases if c["round"] == args.round]
    if args.limit:
        cases = cases[: args.limit]

    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # n, base_n, chars, base_chars, drift
    rows = []
    for i, c in enumerate(cases, 1):
        try:
            r = post({"user_id": c["user_id"], "query": c["query"], "top_k": c["top_k"]})
        except Exception as e:
            print(f"[{i}/{len(cases)}] ERROR {e}")
            continue
        data = r.get("data", [])
        ids = [x.get("id") for x in data]
        chars = sum(len(x.get("content") or "") for x in data)
        drift = len(set(ids[:10]) - set(c["base_top_ids"]))
        a = agg[c["round"]]
        a[0] += len(data); a[1] += c["base_n"] or 0
        a[2] += chars; a[3] += c["base_chars"] or 0
        a[4] += drift
        rows.append({"round": c["round"], "query": c["query"], "n": len(data),
                     "base_n": c["base_n"], "chars": chars,
                     "base_chars": c["base_chars"], "drift": drift, "ids": ids})
        print(f"[{i}/{len(cases)}] {c['round']} n={len(data)}/{c['base_n']} "
              f"chars={chars}/{c['base_chars']} drift={drift} | {c['query'][:48]}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n== 汇总（本轮 vs 官方当时的返回） ==")
    for k in sorted(agg):
        n, bn, ch, bch, dr = agg[k]
        cnt = max(1, sum(1 for row in rows if row["round"] == k))
        print(f"{k}: 平均条数 {n/cnt:.1f}/{bn/cnt:.1f}  "
              f"平均字符 {ch/cnt:.0f}/{bch/cnt:.0f}  头部漂移 {dr/cnt:.1f}/10")


if __name__ == "__main__":
    main()
