# -*- coding: utf-8 -*-
"""对已灌入的 LoCoMo 数据做抽样检索评测。"""
import json
import re

import pandas as pd
from evaluate_locomo import call, norm

df = pd.read_parquet("locomo.parquet")
row = df.iloc[0]
uid = "locomo-diag"

qa_list = json.loads(row["qa"])
per_cat = 5
cat_count = {}
sampled = []
for q in qa_list:
    if q.get("is_adversarial"):
        continue
    cat = q["category"]
    if cat_count.get(cat, 0) < per_cat:
        sampled.append(q)
        cat_count[cat] = cat_count.get(cat, 0) + 1

print(f"抽样 {len(sampled)} 题")
total_hit = 0
total_top1 = 0
cat_stats = {}

for q in sampled:
    cat = q["category"]
    query = q["question"]
    answer = q["answer"]
    s, b = call("POST", "/search", {"query": query, "user_id": uid, "top_k": 10})
    data = b.get("data", [])
    joined = norm(" ".join(it["content"] for it in data))
    top1 = norm(data[0]["content"]) if data else ""
    kws = [norm(w) for w in re.findall(r"\w+", answer) if len(w) > 2]
    hit = all(k in joined for k in kws) if kws else False
    t1 = any(k in top1 for k in kws) if kws else False
    total_hit += 1 if hit else 0
    total_top1 += 1 if t1 else 0
    d = cat_stats.setdefault(cat, {"q": 0, "hit": 0, "top1": 0})
    d["q"] += 1
    d["hit"] += 1 if hit else 0
    d["top1"] += 1 if t1 else 0
    mark = "OK " if hit else "   "
    yn = "Y" if hit else "N"
    yt = "Y" if t1 else "N"
    print(f"[{mark}] cat{cat} {query[:40]:42s} hit={yn} top1={yt}")

print()
print(f"总证据命中率: {total_hit}/{len(sampled)} = {total_hit/len(sampled):.1%}")
print(f"Top1 命中率:  {total_top1}/{len(sampled)} = {total_top1/len(sampled):.1%}")
cat_names = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}
for c in sorted(cat_stats):
    d = cat_stats[c]
    nm = cat_names.get(c, f"cat{c}")
    print(f"{nm:12s}: n={d['q']} hit={d['hit']/d['q']:.1%} top1={d['top1']/d['q']:.1%}")
