#!/usr/bin/env python3
"""P0：id-gold 评测器（替代正则口径）。

与 probe.py 的区别：
  - 命中判定基于 gold_ids（记忆 id），干扰记忆无法「假装命中」
  - 按 gold 条数分桶（窄 / 中 / 宽）报告，避免宽 gold 项虚高
  - coverage 类项（F1 摘要）按 coverage 正则覆盖率计分，不参与 hit 统计
  - 输出 p50/p95 延迟、弃答率（返回 0 条）

用法:
  python3 probe2.py --set sets/eval_v2.json --base http://127.0.0.1:8061 --out results/x1_orig.json
  python3 probe2.py --set sets/neg_v2.json --base http://127.0.0.1:8061 --out results/x1_neg.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request

NARROW_MAX = 5
MID_MAX = 20


def post_search(base: str, user_id: str, query: str, top_k: int, rid: str,
                timeout: float = 300.0) -> tuple[list[dict], float]:
    payload = {"user_id": user_id, "query": query, "top_k": int(top_k), "request_id": rid}
    req = urllib.request.Request(
        base.rstrip("/") + "/search",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return (body.get("data") or []), time.time() - t0


def pct(sel: list[dict], key: str, k: int) -> float:
    return round(sum(1 for r in sel if 0 < r.get(key, 0) <= k) / len(sel), 3)


def mrr(sel: list[dict], key: str = "rank") -> float:
    return round(sum(1.0 / r[key] for r in sel if r.get(key)) / len(sel), 4)


def block(sel: list[dict], key: str = "rank") -> dict:
    if not sel:
        return {}
    lat = sorted(r["elapsed"] for r in sel)
    return {
        "n": len(sel),
        "hit@1": pct(sel, key, 1),
        "hit@5": pct(sel, key, 5),
        "hit@10": pct(sel, key, 10),
        "mrr": mrr(sel, key),
        "avg_n": round(statistics.mean(r["n"] for r in sel), 1),
        "p50_s": round(lat[len(lat) // 2], 2),
        "p95_s": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2),
        "abstain_rate": round(sum(1 for r in sel if r["n"] == 0) / len(sel), 3),
    }


def bucket(gc: int) -> str:
    return "narrow" if gc <= NARROW_MAX else ("mid" if gc <= MID_MAX else "wide")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    spec = json.load(open(args.set, encoding="utf-8"))
    items = spec["items"] if isinstance(spec, dict) else spec

    rows = []
    for n, it in enumerate(items):
        try:
            docs, elapsed = post_search(
                args.base, it["user_id"], it["query"], it.get("top_k", 100), f"p2-{n}"
            )
            err = ""
        except Exception as exc:  # noqa: BLE001
            docs, elapsed, err = [], 0.0, f"{type(exc).__name__}: {exc}"
        ids = [d.get("id") for d in docs]
        contents = [(d.get("content") or "") for d in docs]
        gold = set(it.get("gold_ids") or [])
        rank = next((i for i, did in enumerate(ids, 1) if did in gold), 0)
        row = {
            "id": it["id"], "cat": it.get("cat"), "variant": it.get("variant"),
            "gold_count": it.get("gold_count", len(gold)), "n": len(docs),
            "rank": rank, "elapsed": round(elapsed, 3),
        }
        if err:
            row["error"] = err
        cov = it.get("coverage") or []
        if cov:
            blob = "\n".join(contents)
            hit_cov = [p for p in cov if re.search(p, blob, re.IGNORECASE)]
            row["gold_bucket"] = "coverage"
            row["cover_ratio"] = round(len(hit_cov) / len(cov), 3)
            row["cover_miss"] = [p for p in cov if p not in hit_cov]
        else:
            row["gold_bucket"] = bucket(row["gold_count"])
        if it.get("negative"):
            probe = it.get("probe") or ""
            row["gold_bucket"] = "negative"
            row["probe"] = probe
            row["n_with_probe"] = sum(1 for c in contents if probe and probe in c)
        rows.append(row)

    fact = [r for r in rows if r["gold_bucket"] in ("narrow", "mid", "wide")]
    covs = [r for r in rows if r["gold_bucket"] == "coverage"]
    negs = [r for r in rows if r["gold_bucket"] == "negative"]

    summary: dict = {"tag": args.tag, "all": block(fact)}
    for b in ("narrow", "mid", "wide"):
        summary[b] = block([r for r in fact if r["gold_bucket"] == b])
    for v in sorted({r["variant"] for r in fact if r["variant"]}):
        summary["var_" + str(v)] = block([r for r in fact if r["variant"] == v])
    for c in sorted({r["cat"] for r in fact if r["cat"]}):
        summary["cat_" + str(c)] = block([r for r in fact if r["cat"] == c])
    if covs:
        summary["coverage"] = {
            "n": len(covs),
            "cover_ratio": round(statistics.mean(r["cover_ratio"] for r in covs), 3),
            "cover_full": round(sum(1 for r in covs if r["cover_ratio"] == 1.0) / len(covs), 3),
            "avg_n": round(statistics.mean(r["n"] for r in covs), 1),
            "p95_s": round(sorted(r["elapsed"] for r in covs)[-1], 2),
        }
    if negs:
        summary["negative"] = {
            "n": len(negs),
            "avg_n": round(statistics.mean(r["n"] for r in negs), 1),
            "probe_hit_rate": round(sum(1 for r in negs if r["n_with_probe"] > 0) / len(negs), 3),
            "abstain_rate": round(sum(1 for r in negs if r["n"] == 0) / len(negs), 3),
        }

    json.dump({"summary": summary, "rows": rows}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[{args.tag}] {args.set} -> {args.out}")
    for k, v in summary.items():
        if k == "tag":
            continue
        if "hit@1" in v:
            print(f"  {k:22s} n={v['n']:3d} hit@1={v['hit@1']:.3f} hit@10={v['hit@10']:.3f} "
                  f"mrr={v['mrr']:.3f} p95={v['p95_s']}s")
        else:
            print(f"  {k:22s} {json.dumps(v, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
