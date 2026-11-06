#!/usr/bin/env python3
"""泛化性/持久性探针（只读，不改服务）。

用法:
  # 打一组 query，保存每个 query 的返回 id / 排名 / 延迟
  python3 probe.py --set eval_set.json --base http://127.0.0.1:8046 --out /tmp/rb/s0_orig.json
  # 与基线结果对比：受限命中率（只统计基线里出现过的记忆 id）+ top-k 保留率
  python3 probe.py --set eval_set.json --base http://127.0.0.1:8047 \
      --baseline /tmp/rb/s0_orig.json --out /tmp/rb/s1_orig.json
  # 不可回答探测（负样本）
  python3 probe.py --set negative_set.json --base http://127.0.0.1:8047 --out /tmp/rb/s1_neg.json

打分口径:
  raw hit@k  : 直接看返回里第 k 条前是否有满足 gold 正则的记忆（干扰记忆也可能命中，会虚高）
  res hit@k  : 只在「基线跑出来的记忆 id」里算首次命中排名 —— 衡量原答案是否被挤下去
  retention  : 新返回的 top-k 里还保留了多少基线的 top-k 记忆
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request

USERS = {
    "knee": "u_56d3c8e3472cb8d469229c64f3234a2b714d0ca1dfe28d575e95e956305429c9",
}


def post_search(base: str, user_id: str, query: str, top_k: int, timeout: float = 180.0) -> dict:
    payload = {
        "user_id": user_id,
        "query": query,
        "top_k": int(top_k),
        "request_id": f"rb-{int(time.time() * 1000)}-{abs(hash(query)) % 100000}",
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


def doc_ok(content: str, must: list[str], any_: list[str]) -> bool:
    for pat in must or []:
        if not re.search(pat, content, re.IGNORECASE):
            return False
    if any_ and not any(re.search(p, content, re.IGNORECASE) for p in any_):
        return False
    return True


def first_rank(docs: list[dict], must, any_, allowed_ids: set[str] | None) -> int:
    for i, d in enumerate(docs, start=1):
        if allowed_ids is not None and d.get("id") not in allowed_ids:
            continue
        if doc_ok(d.get("content") or "", must, any_):
            return i
    return 0


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}

    def pct(key, k):
        return round(sum(1 for r in rows if 0 < r.get(key, 0) <= k) / n, 3)

    t = sorted(r["elapsed"] for r in rows)
    out = {
        "n": n,
        "raw_hit@1": pct("raw_rank", 1),
        "raw_hit@5": pct("raw_rank", 5),
        "raw_hit@10": pct("raw_rank", 10),
        "raw_mrr": round(sum(1.0 / r["raw_rank"] if r["raw_rank"] else 0.0 for r in rows) / n, 4),
        "res_hit@1": pct("res_rank", 1),
        "res_hit@5": pct("res_rank", 5),
        "res_hit@10": pct("res_rank", 10),
        "res_mrr": round(sum(1.0 / r["res_rank"] if r["res_rank"] else 0.0 for r in rows) / n, 4),
        "avg_n": round(statistics.mean(r["n_returned"] for r in rows), 1),
        "avg_s": round(statistics.mean(t), 2),
        "p95_s": round(t[max(0, int(n * 0.95) - 1)], 2),
    }
    ret = [r for r in rows if "retention@10" in r]
    if ret:
        out["retention@10"] = round(sum(r["retention@10"] for r in ret) / len(ret), 3)
        out["retention@100"] = round(sum(r["retention@100"] for r in ret) / len(ret), 3)
        out["top1_kept"] = round(sum(1 for r in ret if r["top1_kept"]) / len(ret), 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="", help="基线 probe 结果 json，用于受限命中与保留率")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    spec = json.load(open(args.set, encoding="utf-8"))
    base_ids: dict[str, list[str]] = {}
    if args.baseline:
        b = json.load(open(args.baseline, encoding="utf-8"))
        base_ids = {it["id"]: it["ids"] for it in b["items"]}

    rows = []
    for it in spec["items"]:
        uid = USERS.get(it["user"], it["user"])
        try:
            r = post_search(args.base, uid, it["query"], it["top_k"], timeout=args.timeout)
            docs = r["resp"].get("data") or []
            elapsed = r["elapsed"]
        except Exception as exc:  # noqa: BLE001
            docs, elapsed = [], 0.0
            row = {"id": it["id"], "query": it["query"], "error": f"{type(exc).__name__}: {exc}"}
        else:
            row = {"id": it["id"], "query": it["query"]}

        ids = [d.get("id") for d in docs]
        contents = [d.get("content") or "" for d in docs]
        row.update({
            "n_returned": len(docs),
            "n_chars": sum(len(c) for c in contents),
            "elapsed": round(elapsed, 3),
            "ids": ids,
        })

        if it.get("negative"):
            probe = it.get("probe") or ""
            row["probe"] = probe
            row["n_with_probe"] = sum(1 for c in contents if probe and probe in c)
        else:
            allowed = set(base_ids[it["id"]]) if it["id"] in base_ids else None
            row["raw_rank"] = first_rank(docs, it.get("must"), it.get("any"), None)
            row["res_rank"] = first_rank(docs, it.get("must"), it.get("any"), allowed)
            if it["id"] in base_ids:
                b_ids = base_ids[it["id"]]
                b10, b100 = b_ids[:10], b_ids[:100]
                row["retention@10"] = round(len(set(ids[:10]) & set(b10)) / len(b10), 3) if b10 else 0.0
                row["retention@100"] = round(len(set(ids[:100]) & set(b100)) / len(b100), 3) if b100 else 0.0
                row["top1_kept"] = bool(b_ids and ids and ids[0] == b_ids[0])
        rows.append(row)

    if any(it.get("negative") for it in spec["items"]):
        neg = [r for r in rows if "probe" in r]
        summary = {
            "n": len(neg),
            "avg_n_returned": round(statistics.mean(r["n_returned"] for r in neg), 1),
            "avg_n_with_probe": round(statistics.mean(r["n_with_probe"] for r in neg), 2),
            "n_no_probe": sum(1 for r in neg if r["n_with_probe"] == 0),
        }
    else:
        summary = agg(rows)

    out = {"base": args.base, "set": args.set, "summary": summary, "items": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n[{args.base}] {args.set}")
    print(f"{'item':22} {'raw':>6} {'res':>6} {'ret@10':>7} {'ret@100':>8} {'n':>5} {'s':>6}")
    for r in rows:
        if "probe" in r:
            print(f"{r['id']:22} {'-':>6} {'-':>6} {'-':>7} {'-':>8} {r['n_returned']:>5} {r['elapsed']:>6.2f}")
        else:
            print(f"{r['id']:22} {r['raw_rank'] or '-':>6} {r['res_rank'] or '-':>6} "
                  f"{r.get('retention@10', '-'):>7} {r.get('retention@100', '-'):>8} "
                  f"{r['n_returned']:>5} {r['elapsed']:>6.2f}")
    print("summary:", json.dumps(summary, ensure_ascii=False))
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
