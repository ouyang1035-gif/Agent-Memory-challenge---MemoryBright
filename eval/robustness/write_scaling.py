#!/usr/bin/env python3
"""写入(持久性)测压：向同一 user 追加固定批次，测 /add 延迟随库规模的漂移。
s0(789 条) 与 s3(20,853 条) 使用完全相同 add 侧配置，可直接对比。"""
import json, re, statistics, sys, time, urllib.request

KNEE = "u_56d3c8e3472cb8d469229c64f3234a2b714d0ca1dfe28d575e95e956305429c9"
TEXTS = [
    f"写压测试第{i}条：今天下午三点我在康复中心做了第{i}组膝关节屈伸训练，角度提升到{60+i%20}度，训练后冰敷十分钟。"
    for i in range(20)
]


def post_add(base, rid, idx):
    payload = {
        "user_id": KNEE,
        "session_id": f"writetest-{idx}",
        "request_id": rid,
        "messages": [{"role": "user", "content": t, "timestamp": 1745000000000 + idx * 1000 + i}
                     for i, t in enumerate(TEXTS)],
    }
    req = urllib.request.Request(base.rstrip("/") + "/add", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.time() - t0


def main():
    arm = sys.argv[1]
    base = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    lat = []
    for i in range(n):
        dt = post_add(base, f"writetest-{arm}-{i}", i)
        lat.append(dt)
        print(f"{arm} add#{i} {dt:.2f}s", flush=True)
    print(json.dumps({"arm": arm, "n": n, "mean_s": round(statistics.mean(lat), 2),
                      "median_s": round(statistics.median(lat), 2),
                      "max_s": round(max(lat), 2), "min_s": round(min(lat), 2)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
