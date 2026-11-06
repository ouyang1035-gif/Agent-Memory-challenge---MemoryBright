#!/usr/bin/env python3
"""构造"记忆量放大"实验：往 knee 用户里灌入同类/无关的历史记忆。

两类干扰：
  near  : 从 knee 真实的 fact 句子改写而来（同模板、同实体，只改数字/日期）——最难
  synth : 合成的临床/摄影日记体句子，一半同主题、一半无关

数字取值刻意避开 5 / 10 / 15 / 20（即评测集 gold 的数值），避免干扰记忆本身命中 gold 正则。

用法:
  python3 distractors.py --base http://127.0.0.1:8047 --store /tmp/rb_s1/memory.db \
      --near 5000 --synth 0
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import time
import urllib.request

USER_ID = "u_56d3c8e3472cb8d469229c64f3234a2b714d0ca1dfe28d575e95e956305429c9"
NUM_CHOICES = [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 22, 23, 25, 30, 35, 40]
CN_CHOICES = ["两", "三", "四", "六", "七", "八", "九", "十一", "十二", "十三", "十四", "十六", "十八", "二十几", "三十几"]
CN_RE = re.compile(r"(?:十[一二三四五六七八九]?|二十[一二三四五六七八九]?|[一二三四五六七八九])(?=分钟|小时|天|公斤|次|周|个月)")
DATE_RE = re.compile(r"20\d{2}[-年]\d{1,2}[-月]\d{1,2}日?")
DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")

SYNTH_SAME = [
    "2025-{m:02d}-{d:02d} 外拍一整天，走了{walk}公里，背{kg}公斤器材，晚上右膝酸胀，NRS {nrs} 分。",
    "2025-{m:02d}-{d:02d} 起床时右膝僵硬约{dur}分钟，活动一会儿才缓过来。",
    "2025-{m:02d}-{d:02d} 去社区医院复诊，医生让我继续用{med}，再观察一周。",
    "2025-{m:02d}-{d:02d} 前晚下雨，右膝有点发紧，热敷{dur}分钟后舒服一些。",
    "2025-{m:02d}-{d:02d} 拍婚礼站了一整天，回来右膝外侧有点刺痛，休息后减轻。",
    "2025-{m:02d}-{d:02d} 早上晨僵大约{dur}分钟，下楼时膝盖前面偏内侧不太舒服。",
    "2025-{m:02d}-{d:02d} 涂了{med}以后皮肤有点发热，过一阵就退了。",
    "2025-{m:02d}-{d:02d} 试了新的护膝，走{walk}公里后膝盖负担感比之前轻一些。",
]
SYNTH_OTHER = [
    "2025-{m:02d}-{d:02d} 新镜头到了，试拍了一下午，主要拍建筑和街景。",
    "2025-{m:02d}-{d:02d} 修图到深夜，第二天精神不太好，睡了{walk}小时。",
    "2025-{m:02d}-{d:02d} 和客户沟通拍摄方案，确认了场地和服装。",
    "2025-{m:02d}-{d:02d} 整理去年的作品集，挑出{walk}组照片准备投稿。",
    "2025-{m:02d}-{d:02d} 天气很好，去公园散步{dur}分钟，顺便拍了几张风景。",
    "2025-{m:02d}-{d:02d} 买了新的相机包，可以装{kg}公斤装备，背起来轻松不少。",
    "2025-{m:02d}-{d:02d} 晚上煮了面，看了会儿纪录片，{walk}点就睡了。",
    "2025-{m:02d}-{d:02d} 给朋友修了一组旅拍照片，他请我吃了顿饭。",
]
MEDS = ["扶他林", "塞来昔布", "双氯芬酸凝胶", "外用消炎膏"]


def mutate(text: str, rng: random.Random) -> str:
    def repl_num(m: re.Match) -> str:
        return str(rng.choice(NUM_CHOICES))

    def repl_date(m: re.Match) -> str:
        return f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"

    def repl_cn(m: re.Match) -> str:
        return rng.choice(CN_CHOICES)

    out = DATE_RE.sub(repl_date, text)
    out = DIGIT_RE.sub(repl_num, out)
    out = CN_RE.sub(repl_cn, out)
    return out


def build_near(store: str, n: int, rng: random.Random) -> list[str]:
    con = sqlite3.connect(store)
    facts = [r[0] for r in con.execute(
        "select content from memories where user_id=? and kind='fact'", (USER_ID,)
    )]
    con.close()
    if not facts:
        raise SystemExit("no facts found")
    out, seen = [], set()
    rounds = 0
    while len(out) < n:
        for f in facts:
            if len(out) >= n:
                break
            t = mutate(f, rng)
            if t in seen or t == f:
                continue
            seen.add(t)
            out.append(t)
        rounds += 1
        if rounds > 200:
            break
    return out


def build_synth(n: int, rng: random.Random) -> list[str]:
    out, seen = [], set()
    while len(out) < n:
        tpl = rng.choice(SYNTH_SAME if rng.random() < 0.5 else SYNTH_OTHER)
        t = tpl.format(
            m=rng.randint(1, 12),
            d=rng.randint(1, 28),
            walk=rng.choice([3, 4, 6, 7, 8, 9, 11, 12]),
            kg=rng.choice([2, 3, 4, 6, 7, 8, 9]),
            nrs=rng.choice(NUM_CHOICES),
            dur=rng.choice([2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]),
            med=rng.choice(MEDS),
        )
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def post_add(base: str, texts: list[str], tag: str, idx: int) -> None:
    payload = {
        "request_id": f"distract-{tag}-{idx:06d}",
        "user_id": USER_ID,
        "session_id": f"distract-{tag}",
        "messages": [
            {"role": "user", "content": t, "timestamp": 1740000000000 + idx * 1000 + i}
            for i, t in enumerate(texts)
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/add", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        resp.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--near", type=int, default=0)
    ap.add_argument("--synth", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    texts = build_near(args.store, args.near, rng) + build_synth(args.synth, rng)
    rng.shuffle(texts)
    print(f"generated {len(texts)} distractor memories")
    for t in texts[:3]:
        print("  sample:", t)

    t0 = time.time()
    for i in range(0, len(texts), args.batch):
        chunk = texts[i:i + args.batch]
        post_add(args.base, chunk, f"{args.seed}", i // args.batch)
        if (i // args.batch + 1) % 25 == 0:
            done = i + len(chunk)
            print(f"  added {done}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(texts)} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
