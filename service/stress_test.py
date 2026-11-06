# -*- coding: utf-8 -*-
"""高难度压力测试：暴露检索瓶颈。

场景：一个用户有 ~60 条记忆（大量日常闲聊噪声 + 少量关键事实），
问题需要从大噪声中精准召回。同时测响应时间（Full 评测有并发和超时要求）。
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(method, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def norm(t):
    return t.replace(" ", "").lower()


def build_noise_dialogue(user_id):
    """构造一个含大量噪声的长对话历史。"""
    convs = []

    # 关键事实（散落在大量噪声中）
    key_facts = [
        ("我 2022 年 5 月买了人生第一辆车，是辆白色的特斯拉 Model 3", 1651334400000),
        ("上个月我把车换成了比亚迪汉，白色的特斯拉卖掉了", 1704672000000),
        ("我最爱的编程语言其实是 Rust，虽然工作用 Go", 1704240000000),
        ("我家猫叫咪咪，是一只橘猫，特别能吃", 1704153600000),
        ("我计划 2025 年 3 月去日本看樱花", 1704067200000),
    ]

    # 大量闲聊噪声
    noise_topics = [
        "今天天气不错", "中午吃了黄焖鸡", "地铁上人好多", "昨天看了个电影一般般",
        "楼下新开了家奶茶店", "周末打算睡个懒觉", "手机快没电了", "快递到了",
        "今天开了三个会好累", "晚上想吃火锅", "这件衣服挺好看的", "最近胖了",
        "天气变冷了要加衣服", "咖啡喝多了睡不着", "今天堵车堵了半小时",
        "同事分享了个搞笑的段子", "网上买了个新键盘", "这个剧追完了",
        "周末去爬了山", "今天心情不错", "午饭吃了沙拉", "想换个发型",
    ]

    ts = 1650000000000
    conv_idx = 0

    # 交织关键事实和噪声
    noise_i = 0
    for fact_text, fact_ts in key_facts:
        # 每个关键事实前后插入噪声
        block = []
        for _ in range(4):  # 4 条噪声
            block.append({"role": "user", "content": noise_topics[noise_i % len(noise_topics)], "timestamp": ts})
            noise_i += 1
            ts += 60000
            block.append({"role": "assistant", "content": "嗯嗯，好的", "timestamp": ts})
            ts += 60000
        # 插入关键事实
        block.append({"role": "user", "content": fact_text, "timestamp": fact_ts})
        block.append({"role": "assistant", "content": "明白了，记下来了", "timestamp": fact_ts + 1000})
        convs.append((f"{user_id}-stress-{conv_idx}", block))
        conv_idx += 1

    # 补齐剩余噪声
    while noise_i < len(noise_topics) * 2:
        block = [
            {"role": "user", "content": noise_topics[noise_i % len(noise_topics)], "timestamp": ts},
            {"role": "assistant", "content": "好的", "timestamp": ts + 1000},
        ]
        convs.append((f"{user_id}-stress-{conv_idx}", block))
        noise_i += 1
        conv_idx += 1
        ts += 120000

    return convs


STRESS_QUESTIONS = [
    {"query": "用户现在开什么车？", "must_have": ["比亚迪"], "note": "需冲突消解（特斯拉已卖）"},
    {"query": "用户最爱的编程语言？", "must_have": ["Rust"], "note": "注意区分工作用 Go"},
    {"query": "用户的猫叫什么、什么品种？", "must_have": ["咪咪", "橘猫"], "note": "列表型"},
    {"query": "用户计划什么时候去哪里旅游？", "must_have": ["2025", "日本"], "note": "时间+地点"},
    {"query": "用户第一辆车是什么颜色什么型号？", "must_have": ["白色", "Model 3"], "note": "历史事实，不能因换车而答错"},
]


def main():
    uid = "eval-stress"
    convs = build_noise_dialogue(uid)
    total_msgs = sum(len(c[1]) for c in convs)
    print(f"准备写入 {len(convs)} 个对话块，共约 {total_msgs} 条消息（含大量噪声）...")

    t0 = time.time()
    for rid, block in convs:
        s, b = call("POST", "/add", {
            "request_id": rid, "messages": block,
            "user_id": uid, "session_id": f"{uid}-s",
        })
        if s != 200:
            print(f"[ERROR] add {rid}: {s} {b}")
    add_time = time.time() - t0
    print(f"写入完成，耗时 {add_time:.1f}s\n")

    print("===== 检索压测 =====")
    total_r = 0.0
    total_t = 0.0
    for q in STRESS_QUESTIONS:
        t1 = time.time()
        s, b = call("POST", "/search", {"query": q["query"], "user_id": uid, "top_k": 10})
        dt = time.time() - t1
        total_t += dt
        data = b.get("data", [])
        joined = norm(" ".join(it["content"] for it in data))
        top1 = norm(data[0]["content"]) if data else ""
        hit = sum(1 for k in q["must_have"] if norm(k) in joined)
        r = hit / len(q["must_have"])
        total_r += r
        top1_ok = any(norm(k) in top1 for k in q["must_have"])
        print(f"[{'OK ' if r==1.0 and top1_ok else '   '}] {q['query'][:20]:22s} R={r:.2f} top1={'Y' if top1_ok else 'N'} {dt:.2f}s | {q['note']}")
        if not top1_ok and data:
            print(f"        top1实际返回: {data[0]['content'][:40]}")

    n = len(STRESS_QUESTIONS)
    print(f"\n平均证据召回率: {total_r/n:.2%}")
    print(f"平均检索耗时:   {total_t/n:.2f}s")
    print(f"记忆规模:       ~{total_msgs} 条消息")


if __name__ == "__main__":
    main()
