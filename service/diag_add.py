# -*- coding: utf-8 -*-
"""诊断脚本：小批量灌入 LoCoMo，定位服务崩溃原因。"""
import json
import time
import urllib.request
import urllib.error

import pandas as pd
from evaluate_locomo import parse_context

BASE = "http://127.0.0.1:8000"


def call(method, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")[:500]
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


# 先确认服务活着
s, b = call("GET", "/health")
print(f"health: {s} {b}")
if s != 200:
    print("服务未启动，退出")
    raise SystemExit(1)

# 解析第一个 sample
df = pd.read_parquet("locomo.parquet")
row = df.iloc[0]
messages = parse_context(row["context"], row["speaker_a"], row["speaker_b"])
print(f"解析出 {len(messages)} 条消息")

# 逐批灌入，每批后检查服务是否还活着
batch_size = 20
uid = "locomo-diag"
for i in range(0, len(messages), batch_size):
    batch = messages[i:i + batch_size]
    bn = i // batch_size
    t0 = time.time()
    s, b = call("POST", "/add", {
        "request_id": f"{uid}-b{bn}",
        "messages": batch,
        "user_id": uid,
        "session_id": f"{uid}-s{bn}",
    })
    dt = time.time() - t0
    print(f"batch {bn:2d} ({len(batch)} msgs): status={s} time={dt:.1f}s", 
          "" if s == 200 else f"ERROR: {b}")
    if s != 200:
        print(f"!!! 第 {bn} 批失败，服务可能已崩溃")
        # 检查服务
        hs, hb = call("GET", "/health")
        print(f"    health check: {hs}")
        break
print("灌入测试完成")
