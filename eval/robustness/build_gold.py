#!/usr/bin/env python3
"""P0：把 eval_set 的「内容正则 gold」升级为「记忆 id gold」。

原因：正则 gold 在加了干扰记忆的库里会假命中（干扰文本碰巧含 gold 词），
导致规模越大越容易「看起来没退化」。改成 id 集合后，命中必须是返回了
真正的答案记忆。

用法:
  python3 build_gold.py --set /workspace/amc_eval/eval_set.json \
      --store /workspace/amc_eval/eval_base_store/memory.db \
      --out sets/eval_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3

USERS = {
    "knee": "u_56d3c8e3472cb8d469229c64f3234a2b714d0ca1dfe28d575e95e956305429c9",
}


def doc_ok(content: str, must: list[str], any_: list[str]) -> bool:
    for pat in must or []:
        if not re.search(pat, content, re.IGNORECASE):
            return False
    if any_ and not any(re.search(p, content, re.IGNORECASE) for p in any_):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="/workspace/amc_eval/eval_set.json")
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--para", default="", help="改写集路径；gold 由 src 指向的原项继承")
    args = ap.parse_args()

    spec = json.load(open(args.set, encoding="utf-8"))
    items = spec["items"] if isinstance(spec, dict) else spec
    users = dict(USERS)
    if isinstance(spec, dict):
        users.update(spec.get("users") or {})

    con = sqlite3.connect(f"file:{args.store}?mode=ro", uri=True)
    cur = con.cursor()

    out_items, skipped = [], []
    for it in items:
        uid = users[it["user"]]
        rows = cur.execute(
            "select id, content from memories where user_id=?", (uid,)
        ).fetchall()
        gold = [rid for rid, content in rows if doc_ok(content or "", it.get("must"), it.get("any"))]
        new = dict(it)
        new["user_id"] = uid
        new["gold_ids"] = sorted(gold)
        new["gold_count"] = len(gold)
        new["kind"] = "fact" if gold else "coverage"
        if not gold:
            skipped.append(it["id"])
        out_items.append(new)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"_doc": "gold = 记忆 id 集合（由内容正则一次性物化）", "items": out_items},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"items={len(out_items)} fact={sum(1 for i in out_items if i['kind']=='fact')} "
          f"coverage/no-gold={len(skipped)} {skipped}")

    if args.para:
        by_id = {i["id"]: i for i in out_items}
        pspec = json.load(open(args.para, encoding="utf-8"))
        pitems = pspec["items"] if isinstance(pspec, dict) else pspec
        pout, missing = [], []
        for p in pitems:
            src = by_id.get(p.get("src"))
            if src is None:
                missing.append(p.get("id"))
                continue
            q = dict(p)
            q["user_id"] = src["user_id"]
            q["gold_ids"] = src["gold_ids"]
            q["gold_count"] = src["gold_count"]
            q["kind"] = src["kind"]
            q["cat"] = src.get("cat") or src.get("cat")
            pout.append(q)
        para_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "para_v2.json")
        json.dump({"_doc": "改写集 gold 继承自 src 原项", "items": pout},
                  open(para_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"para items={len(pout)} missing_src={missing} -> {para_path}")

    for i in out_items[:5]:
        print(f"  {i['id']}: gold={i['gold_count']} ids={i['gold_ids'][:3]}")


if __name__ == "__main__":
    main()
