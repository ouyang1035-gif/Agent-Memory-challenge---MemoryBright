#!/usr/bin/env python3
"""构建扩样回归集（第 2 层）：LoCoMo-MC10 抽样 + evidence 级 id-gold。

gold 来源（三级，优先级从高到低）
1. evidence（原始 LoCoMo 的 dia_id，如 D8:6）：取该轮对话原文，在冻结库里定位
   包含它的 chunk 记忆；再把 source_id 指向这些 chunk 的 fact 记忆一并纳入 gold。
   —— 这是最可靠的 gold，等价于「答案出处」。
2. 无 evidence 或定位失败：回退到答案关键词全命中的记忆（kind=weak）。
3. 仍无：kind=coverage，只参与 aic 类指标，不参与 hit@k。

规则
- 每 conv 取 N 题，四类题型轮转；排除 adversarial（无 gold、考弃答）。
- top_k 固定 100，与官方 smoke 口径一致。
- 只读冻结库（expanded_store），只做 gold 物化，不写入。

用法:
  python3 build_set.py --per-conv 20 --store /workspace/amc_eval/expanded_store/memory.db \
      --out sets/mc10_200.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter

BENCH = ("/workspace/user_data/agent-memory/Agent-Memory-Challenge-server-20260929/"
         "Agent-Memory-Challenge/memory-service/bench")
sys.path.insert(0, BENCH)
from adapters import DATASET_DIR, STOPWORDS  # noqa: E402

MC10 = os.path.join(DATASET_DIR, "locomo_mc10.json")
RAW = os.path.join(DATASET_DIR, "locomo10_raw.json")
TYPES = ["single_hop", "multi_hop", "temporal_reasoning", "open_domain"]
_DIA_RE = re.compile(r"^D(\d+):(\d+)$")
_TOK_RE = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def tokens(text: str, strict: bool = True) -> list[str]:
    out, seen = [], set()
    for t in _TOK_RE.findall((text or "").lower()):
        if t.isdigit():
            t = str(int(t))
        elif strict and (len(t) < 3 or t in STOPWORDS):
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def load_raw() -> dict:
    """sample_id -> {"dia": {dia_id: text}, "qa": [...]}。"""
    raw = json.load(open(RAW, encoding="utf-8"))
    out = {}
    for s in raw:
        dia = {}
        for k, v in s["conversation"].items():
            if not k.startswith("session_") or k.endswith("_date_time"):
                continue
            if not isinstance(v, list):
                continue
            for turn in v:
                did = turn.get("dia_id")
                if did and turn.get("text"):
                    dia[did] = turn["text"]
        out[s["sample_id"]] = {"dia": dia, "qa": s.get("qa") or []}
    return out


def load_mc10() -> dict:
    convs: dict[str, list[dict]] = {}
    with open(MC10, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            conv = o["question_id"].rsplit("_q", 1)[0]
            o["_idx"] = int(o["question_id"].rsplit("_q", 1)[1])
            convs.setdefault(conv, []).append(o)
    return convs


def sample(convs: dict, per_conv: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    picked = []
    for conv in sorted(convs):
        buckets: dict[str, list[dict]] = {}
        for q in convs[conv]:
            if q.get("question_type") == "adversarial":
                continue
            buckets.setdefault(q["question_type"], []).append(q)
        for b in buckets.values():
            rng.shuffle(b)
        idx, order = 0, []
        while len(order) < per_conv:
            progressed = False
            for t in TYPES:
                b = buckets.get(t) or []
                if idx < len(b):
                    order.append(b[idx])
                    progressed = True
                    if len(order) >= per_conv:
                        break
            if not progressed:
                break
            idx += 1
        picked.extend(order)
    return picked


class Store:
    def __init__(self, path: str):
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self._cache: dict[str, list[dict]] = {}

    def rows(self, uid: str) -> list[dict]:
        if uid not in self._cache:
            out = []
            for rid, sess, kind, content, source_id in self.con.execute(
                "select id, session_id, kind, content, source_id from memories where user_id=?",
                (uid,),
            ):
                out.append({
                    "id": rid, "session": sess, "kind": kind, "norm": norm(content),
                    "toks": set(tokens(content, strict=False)), "source": source_id,
                })
            self._cache[uid] = out
        return self._cache[uid]


def evidence_gold(rows: list[dict], dia_map: dict, evidence: list[str]) -> tuple[list[str], list[str]]:
    chunk_ids, missing = set(), []
    for did in evidence:
        m = _DIA_RE.match(did or "")
        text = dia_map.get(did)
        if not m or not text:
            missing.append(did)
            continue
        sess = "session_%d" % int(m.group(1))
        ntext = norm(text)
        cands = [r for r in rows if r["kind"] == "chunk" and r["session"] == sess
                 and ntext and ntext in r["norm"]]
        if not cands:  # 跨 chunk 截断时退化为前 10 个 token 的连续片段
            head = " ".join(ntext.split()[:10])
            cands = [r for r in rows if r["kind"] == "chunk" and r["session"] == sess
                     and head and head in r["norm"]]
        if not cands:
            missing.append(did)
            continue
        chunk_ids |= {r["id"] for r in cands}
    fact_ids = {r["id"] for r in rows if r["kind"] == "fact" and r["source"] in chunk_ids}
    return sorted(chunk_ids | fact_ids), missing


def build_item(q: dict, rows: list[dict], dia_map: dict) -> dict:
    evidence = [e for e in (q.get("_evidence") or []) if e]
    gt = tokens(q.get("answer") or "")
    gold, missing = evidence_gold(rows, dia_map, evidence)
    if gold:
        kind = "evidence"
    else:
        gold = [r["id"] for r in rows if gt and all(t in r["toks"] for t in gt)]
        kind = "keyword" if gold else "coverage"
        if not gold:
            gold = [r["id"] for r in rows if gt and any(t in r["toks"] for t in gt)]
            kind = "weak" if gold else "coverage"
    return {
        "id": q["question_id"],
        "conv": q["conv"],
        "user_id": q["user_id"],
        "qtype": q.get("question_type"),
        "query": q.get("question"),
        "answer": q.get("answer"),
        "choices": q.get("choices"),
        "evidence": evidence,
        "evidence_missing": missing,
        "gold_tokens": gt,
        "gold_ids": gold,
        "gold_count": len(gold),
        "gold_kind": kind,
        "top_k": 100,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-conv", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--store", default="/workspace/amc_eval/expanded_store/memory.db")
    ap.add_argument("--out", default="sets/mc10_200.json")
    args = ap.parse_args()

    convs = load_mc10()
    raw = load_raw()
    store = Store(args.store)
    picked = sample(convs, args.per_conv, args.seed)

    items = []
    for q in picked:
        q["conv"] = q["question_id"].rsplit("_q", 1)[0]
        q["user_id"] = "locomo_" + q["conv"]
        # mc10 与原始 qa 顺序严格一致（已校验 400/400）
        qa = raw[q["conv"]]["qa"]
        q["_evidence"] = qa[q["_idx"]].get("evidence") if q["_idx"] < len(qa) else []
        items.append(build_item(q, store.rows(q["user_id"]), raw[q["conv"]]["dia"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"_doc": "扩样回归集：LoCoMo-MC10 抽样，gold=evidence 定位的记忆 id，top_k=100",
               "top_k": 100, "per_conv": args.per_conv, "seed": args.seed,
               "items": items},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"items={len(items)} convs={len(set(i['conv'] for i in items))}")
    print("qtype:", dict(Counter(i["qtype"] for i in items)))
    print("gold_kind:", dict(Counter(i["gold_kind"] for i in items)))
    for k in ("evidence", "keyword", "weak", "coverage"):
        gc = [i["gold_count"] for i in items if i["gold_kind"] == k]
        if gc:
            print(f"  {k}: n={len(gc)} min={min(gc)} med={sorted(gc)[len(gc)//2]} max={max(gc)}")
    print("coverage-only:", [i["id"] for i in items if i["gold_kind"] == "coverage"][:15])


if __name__ == "__main__":
    main()
