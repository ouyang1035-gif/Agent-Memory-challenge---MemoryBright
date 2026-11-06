#!/usr/bin/env python3
"""写入路径对比：给 v1/v2 分别 /add 同一批合成对话，再 /search 验证写侧优化是否生效。

覆盖 v2 的三项写侧改动：
  A. 相对时间 → 绝对日期派生（上个月 → 具体年月）
  B. 事实去重时「数值冲突保留变体」（200mg / 400mg 都留下）
  C. 状态更新（杭州 → 南京）与口水话降权

注意：只写隔离副本（8020/8021）的库，不碰线上 8000。
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import datetime, timezone, timedelta

TARGETS = {"v1": "http://127.0.0.1:8020", "v2": "http://127.0.0.1:8021"}
CST = timezone(timedelta(hours=8))


def ts_ms(y, m, d, hh=10):
    return int(datetime(y, m, d, hh, 0, 0, tzinfo=CST).timestamp() * 1000)


def post(base, path, payload, timeout=300.0):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_sessions():
    """返回 [(user_id, session_id, messages)]，user_id 里带 run tag 由调用方替换。"""
    s = []

    u = "eval_w_derive"
    s.append((u, "s1", [
        {"role": "user", "content": "医生，我上个月刚做过胃镜，结果是慢性浅表性胃炎，还有一点幽门螺杆菌阳性。", "timestamp": ts_ms(2025, 3, 15)},
        {"role": "assistant", "content": "明白了，胃镜提示慢性浅表性胃炎伴幽门螺杆菌阳性，这个后面需要按消化科方案复查。", "timestamp": ts_ms(2025, 3, 15)},
        {"role": "user", "content": "另外我对青霉素过敏，之前吃过一次起了皮疹。", "timestamp": ts_ms(2025, 3, 15)},
        {"role": "assistant", "content": "已记录：青霉素过敏，之后开药请主动告知医生。", "timestamp": ts_ms(2025, 3, 15)},
    ]))

    u = "eval_w_variant"
    s.append((u, "s1", [
        {"role": "user", "content": "我现在每天吃布洛芬，一次200毫克，一天一次。", "timestamp": ts_ms(2025, 3, 16)},
        {"role": "assistant", "content": "好的，布洛芬200毫克每天一次，注意饭后服用。", "timestamp": ts_ms(2025, 3, 16)},
        {"role": "user", "content": "医生让我把布洛芬加到一次400毫克，一天两次。", "timestamp": ts_ms(2025, 3, 16, 11)},
        {"role": "assistant", "content": "收到，布洛芬改为400毫克一天两次，加量后留意胃部不适。", "timestamp": ts_ms(2025, 3, 16, 11)},
    ]))

    u = "eval_w_update"
    s.append((u, "s1", [
        {"role": "user", "content": "我目前住在杭州，工作在滨江。", "timestamp": ts_ms(2025, 3, 17)},
        {"role": "assistant", "content": "了解，你目前常住杭州滨江。", "timestamp": ts_ms(2025, 3, 17)},
        {"role": "user", "content": "我上个月已经搬到南京了，杭州那边房子退了。", "timestamp": ts_ms(2025, 3, 18)},
        {"role": "assistant", "content": "好的，你现在的常住地是南京。", "timestamp": ts_ms(2025, 3, 18)},
    ]))

    u = "eval_w_filler"
    msgs = []
    for i, t in enumerate(["嗯嗯", "好的好的", "谢谢医生", "嗯", "行", "好嘞"]):
        msgs.append({"role": "user", "content": t, "timestamp": ts_ms(2025, 3, 19) + i * 60000})
    msgs.insert(3, {"role": "user", "content": "我对海鲜过敏，吃了会起疹子，尤其是虾。", "timestamp": ts_ms(2025, 3, 19) + 180000})
    msgs.insert(4, {"role": "assistant", "content": "已记录：海鲜（尤其虾）过敏会起疹子。", "timestamp": ts_ms(2025, 3, 19) + 240000})
    s.append((u, "s1", msgs))
    return s


CHECKS = [
    # (name, cat, user, query, top_k, must, any)
    ("W-date-abs", "A/C1", "eval_w_derive", "我是什么时候做的胃镜？", 10, [], ["2025\\s*[-年./]\\s*0?2"]),
    ("W-allergy", "A1", "eval_w_derive", "我对哪种药过敏？", 10, ["青霉素"], []),
    ("W-gastroscopy", "A1", "eval_w_derive", "我的胃镜结果是什么？", 10, ["浅表性胃炎"], []),
    ("W-dosage-now", "D1", "eval_w_variant", "我现在布洛芬一次吃多少毫克？", 10, [], ["400\\s*毫克|400mg"]),
    ("W-dosage-both", "D2", "eval_w_variant", "布洛芬的剂量前后有什么变化？", 10, ["200", "400"], []),
    ("W-city-now", "D1", "eval_w_update", "我现在住在哪个城市？", 10, ["南京"], []),
    ("W-shellfish", "A1", "eval_w_filler", "我吃海鲜会怎么样？", 5, ["海鲜"], []),
]


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%m%d%H%M")
    sessions = build_sessions()

    for name, base in TARGETS.items():
        uid_map = {}
        for i, (uid, sid, msgs) in enumerate(sessions):
            real_uid = f"{uid}_{tag}"
            uid_map[uid] = real_uid
            payload = {"request_id": f"eval-add-{tag}-{name}-{i}", "user_id": real_uid,
                       "session_id": sid, "messages": msgs}
            try:
                r = post(base, "/add", payload)
                print(f"[{name}] add {uid} -> {r.get('success')}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] add {uid} FAILED {type(exc).__name__}: {exc}")
        uid_map_store = uid_map
        if name == "v1":
            globals()["_uid_map_v1"] = uid_map_store
        else:
            globals()["_uid_map_v2"] = uid_map_store

    print("\n" + "=" * 92)
    print("写入路径对比（同一合成对话分别写入 v1/v2）")
    print("=" * 92)
    print(f"{'check':16} {'cat':6} | {'v1 rank':>8} {'v2 rank':>8} | {'v1 top1 摘要':<38} | {'v2 top1 摘要':<38}")
    total = {"v1": 0, "v2": 0}
    for (cname, cat, uid, query, k, must, any_) in CHECKS:
        cells = {}
        for name, base in TARGETS.items():
            real_uid = globals()[f"_uid_map_{name}"][uid]
            try:
                resp = post(base, "/search", {"request_id": f"eval-q-{tag}-{name}-{cname}",
                                              "user_id": real_uid, "query": query, "top_k": k})
                docs = [d.get("content") or "" for d in resp.get("data") or []]
            except Exception as exc:  # noqa: BLE001
                docs = []
                print(f"  {name} search failed: {exc}")
            rank = 0
            any_re = "|".join(any_) if any_ else ""
            for i, c in enumerate(docs, start=1):
                ok = all(re.search(p, c, re.I) for p in must)
                if ok and any_re:
                    ok = bool(re.search(any_re, c, re.I))
                if ok:
                    rank = i
                    break
            cells[name] = {"rank": rank, "top1": (docs[0][:36].replace("\n", " ") if docs else "(空)"),
                           "n": len(docs)}
            total[name] += 1 if rank else 0
        print(f"{cname:16} {cat:6} | {cells['v1']['rank']:>8} {cells['v2']['rank']:>8} | "
              f"{cells['v1']['top1']:<38} | {cells['v2']['top1']:<38}")
        if cells["v1"]["rank"] != cells["v2"]["rank"]:
            print(f"{'':16} {'':6} | 差异> v1第{cells['v1']['rank'] or '-'}名  v2第{cells['v2']['rank'] or '-'}名 "
                  f"(items: v1={cells['v1']['n']}, v2={cells['v2']['n']})")
    print("\n命中项数: v1=%d/%d  v2=%d/%d" % (total["v1"], len(CHECKS), total["v2"], len(CHECKS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
