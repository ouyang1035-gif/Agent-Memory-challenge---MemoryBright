#!/usr/bin/env python3
"""扩样回归集判分器（第 2 层）：只打 /search，不写库。

指标（与 19 题哨兵层同口径 + bench 兼容口径）
- hit@1/3/5/10/20 : 返回序列前 k 条里出现过 gold 记忆 id
- MRR             : 首个 gold 记忆名次的倒数
- retention@10    : 前 10 条覆盖了多少比例的 gold id
- aic             : 答案关键词全部出现在返回正文中（bench 口径，跨库可比）
- kw              : 答案关键词覆盖率
- top1_hit        : 首条正文覆盖 >=50% 答案关键词

用法:
  python3 run_expanded.py --base http://127.0.0.1:8031 --tag baseline_74_77 \
      --out reports/baseline_74_77
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.request
from collections import defaultdict

KS = [1, 3, 5, 10, 20]
_TOK_RE = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def token_set(text: str) -> set[str]:
    out = set()
    for t in _TOK_RE.findall((text or "").lower()):
        out.add(str(int(t)) if t.isdigit() else t)
    return out


def post_search(base: str, payload: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + "/search", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, time.time() - t0


def score_one(item: dict, body: dict, elapsed: float) -> dict:
    data = body.get("data") or []
    ids = [str(d.get("id")) for d in data]
    gold = set(item.get("gold_ids") or [])
    rank = next((i + 1 for i, rid in enumerate(ids) if rid in gold), None) if gold else None

    gold_toks = set(item.get("gold_tokens") or [])
    joined = norm(" ".join(str(d.get("content") or "") for d in data))
    jtoks = token_set(joined)
    top1 = token_set(str(data[0].get("content") or "")) if data else set()
    hit_toks = gold_toks & jtoks
    top1_hit = bool(gold_toks) and len(gold_toks & top1) / len(gold_toks) >= 0.5

    rec = {
        "id": item["id"],
        "conv": item["conv"],
        "qtype": item["qtype"],
        "gold_kind": item.get("gold_kind"),
        "gold_count": item.get("gold_count"),
        "n_returned": len(ids),
        "rank": rank,
        "mrr": (1.0 / rank) if rank else 0.0,
        "aic": bool(gold_toks) and gold_toks <= jtoks,
        "kw": (len(hit_toks) / len(gold_toks)) if gold_toks else None,
        "top1_hit": top1_hit,
        "retention@10": (len(gold & set(ids[:10])) / len(gold)) if gold else None,
        "latency": round(elapsed, 2),
    }
    for k in KS:
        rec[f"hit@{k}"] = bool(rank and rank <= k)
    return rec


def agg(rows: list[dict]) -> dict:
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 4) if vals else None

    scored = [r for r in rows if r["gold_kind"] in ("evidence", "keyword")]
    out = {
        "n": len(rows),
        "n_scored": len(scored),
        "n_gold_hit": sum(1 for r in rows if r["mrr"] > 0),
        "aic": mean([1.0 if r["aic"] else 0.0 for r in rows]),
        "kw": mean([r["kw"] for r in rows]),
        "top1_hit": mean([1.0 if r["top1_hit"] else 0.0 for r in rows]),
        "mrr": mean([r["mrr"] for r in rows]),
        "retention@10": mean([r["retention@10"] for r in rows]),
        "latency": mean([r["latency"] for r in rows]),
    }
    for k in KS:
        out[f"hit@{k}"] = mean([1.0 if r[f"hit@{k}"] else 0.0 for r in rows])
    return out


def breakdown(rows: list[dict]) -> dict:
    by_q, by_c = defaultdict(list), defaultdict(list)
    for r in rows:
        by_q[r["qtype"]].append(r)
        by_c[r["conv"]].append(r)
    return {"by_qtype": {k: agg(v) for k, v in sorted(by_q.items())},
            "by_conv": {k: agg(v) for k, v in sorted(by_c.items())}}


def md_report(tag: str, base: str, overall: dict, bd: dict) -> str:
    def pct(v):
        return "-" if v is None else f"{v * 100:.1f}%"

    lines = [f"# 扩样回归集 {tag}", "",
             f"- 端点: `{base}`", f"- 题数: {overall['n']}（可判 gold {overall['n_scored']}）",
             f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "| 指标 | 值 |", "|---|---|"]
    for k in ("hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "mrr", "retention@10",
              "aic", "kw", "top1_hit", "latency"):
        lines.append(f"| {k} | {pct(overall[k]) if k != 'latency' else f'{overall[k]:.2f}s'} |")
    lines += ["", "## 分题型", "", "| 题型 | n | hit@1 | hit@10 | mrr | aic | kw |", "|---|---|---|---|---|---|---|"]
    for q, a in bd["by_qtype"].items():
        lines.append(f"| {q} | {a['n']} | {pct(a['hit@1'])} | {pct(a['hit@10'])} | {a['mrr']:.3f} | {pct(a['aic'])} | {pct(a['kw'])} |")
    lines += ["", "## 分会话", "", "| conv | n | hit@1 | hit@10 | mrr | aic | kw |", "|---|---|---|---|---|---|---|"]
    for c, a in bd["by_conv"].items():
        lines.append(f"| {c} | {a['n']} | {pct(a['hit@1'])} | {pct(a['hit@10'])} | {a['mrr']:.3f} | {pct(a['aic'])} | {pct(a['kw'])} |")
    return "\n".join(lines) + "\n"


def run_all(items: list[dict], base: str, timeout: float, workers: int,
            top_k: int) -> tuple[list[dict], list[dict]]:
    """并发跑完所有题目（线程池）；单题异常只记录不中断。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows, errors = [], []

    def one(i: int, it: dict) -> dict:
        payload = {
            "user_id": it["user_id"],
            "query": it["query"],
            "top_k": it.get("top_k") or top_k,
            "request_id": f"exp-{it['id']}-{int(time.time() * 1000)}",
        }
        body, elapsed = post_search(base, payload, timeout)
        rec = score_one(it, body, elapsed)
        flag = "HIT" if rec["rank"] else "   "
        print(f"[{i:3d}/{len(items)}] {flag} {it['id']:22s} {it['qtype'][:18]:18s} "
              f"rank={rec['rank']} aic={int(rec['aic'])} {elapsed:.1f}s", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(one, i, it): it for i, it in enumerate(items, 1)}
        done = 0
        for f in as_completed(futs):
            it = futs[f]
            done += 1
            try:
                rows.append(f.result())
            except Exception as e:
                errors.append({"id": it["id"], "error": repr(e)[:200]})
                print(f"[{done:3d}/{len(items)}] ERR {it['id']} {e!r}", flush=True)
    order = {it["id"]: n for n, it in enumerate(items)}
    rows.sort(key=lambda r: order[r["id"]])
    return rows, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="sets/mc10_200.json")
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()

    spec = json.load(open(args.set, encoding="utf-8"))
    items = spec["items"]
    if args.limit:
        items = items[:args.limit]
    out_dir = args.out or os.path.join("reports", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()
    rows, errors = run_all(items, args.base, args.timeout, args.workers,
                           spec.get("top_k") or 100)

    overall = agg(rows)
    bd = breakdown(rows)
    summary = {"tag": args.tag, "base": args.base, "set": args.set,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "wall_sec": round(time.time() - t_start, 1),
               "errors": errors, "overall": overall, **bd}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(rows, open(os.path.join(out_dir, "items_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write(
        md_report(args.tag, args.base, overall, bd))

    print("\n== overall ==")
    for k in ("n", "n_scored", "hit@1", "hit@5", "hit@10", "hit@20", "mrr",
              "retention@10", "aic", "kw", "top1_hit", "latency"):
        print(f"  {k}: {overall[k]}")
    print(f"  errors: {len(errors)}  wall: {summary['wall_sec']}s")


if __name__ == "__main__":
    main()
