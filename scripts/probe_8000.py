#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接打线上 /search 测隧道超时。用法：python probe_8000.py"""

import http.client
import json
import socket
import ssl
import time

BASE = "errfowt2-8000.devbridge-s2.hwtunnel.com"
CLIENT_TIMEOUT = 200  # 故意大于网关超时，否则看不到网关回什么

CASES = [
    ("快样本  预期~7s   ", "u_7b4851fbcdf7b6809e849e9eea2e8ab3055ccd90717cb393d9d1e9b5434527a6",
     "Which language is officially spoken in the country that created the sport associated with Bull Durham?"),
    ("中样本  预期~30s  ", "u_dad8878cff347ca6aeac31b65fdb38b5c5687c339989b0b3ad3871160a8ce244",
     "How might Taylor David integrate extreme sports into his routine for stress relief?"),
    ("慢样本  预期80-110s", "u_dad8878cff347ca6aeac31b65fdb38b5c5687c339989b0b3ad3871160a8ce244",
     "What is the name of the non-profit organization founded by Taylor David on June 28, 2029?"),
]


def post_search(user_id, query, top_k=100):
    body = json.dumps({"user_id": user_id, "query": query, "top_k": top_k}).encode()
    conn = http.client.HTTPSConnection(
        BASE, 443, timeout=CLIENT_TIMEOUT, context=ssl.create_default_context()
    )
    t0 = time.time()
    try:
        conn.request("POST", "/search", body=body,
                     headers={"Content-Type": "application/json",
                              "User-Agent": "probe-8000/1.0"})
        resp = conn.getresponse()
        ttfb = time.time() - t0
        try:
            text = resp.read().decode("utf-8", "replace")
        except Exception as exc:
            return {"status": resp.status, "ttfb": ttfb, "total": time.time() - t0,
                    "body": "", "error": f"读响应中断 {type(exc).__name__}: {exc}"}
        return {"status": resp.status, "ttfb": ttfb, "total": time.time() - t0,
                "body": text, "error": ""}
    except socket.timeout:
        return {"status": None, "ttfb": None, "total": time.time() - t0,
                "body": "", "error": f"客户端等待超时（>{CLIENT_TIMEOUT}s）"}
    except Exception as exc:
        return {"status": None, "ttfb": None, "total": time.time() - t0,
                "body": "", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def verdict(r):
    b = (r["body"] or "").lower()
    if r["status"] == 504:
        return "504 网关上游超时 —— 平台拿不到数据"
    if r["status"] == 502:
        return "502 网关连不上上游（隧道/服务挂了）"
    if r["status"] == 200:
        return "200 正常返回"
    if r["error"]:
        return "异常：" + r["error"]
    return f"status={r['status']}"


def main():
    rows = []
    for label, user, query in CASES:
        print("=" * 72)
        print(f"{label}  top_k=100")
        print(f"query: {query[:80]}")
        print("请求中 ...（慢样本要 1-2 分钟，别关窗口）", flush=True)
        r = post_search(user, query)
        detail = ""
        if r["status"] == 200 and r["body"]:
            try:
                n = len(json.loads(r["body"]).get("data", []))
                detail = f"返回 {n} 条"
            except Exception:
                detail = r["body"][:120]
        elif r["body"]:
            detail = r["body"][:200]
        print(f"HTTP={r['status']}  首字节={_s(r['ttfb'])}  总耗时={_s(r['total'])}  {detail}")
        print(f"结论：{verdict(r)}")
        rows.append((label, r["status"], r["total"], verdict(r)))

    print("\n" + "=" * 72)
    print(f"{'样本':<18}{'HTTP':>6}{'总耗时(s)':>12}  结论")
    for label, status, total, v in rows:
        print(f"{label:<18}{str(status):>6}{total:>12.1f}  {v}")
    print("\n判读：慢样本 504 → 网关确实 >60s 就断；慢样本 200 且 >65s → 之前结论作废。")


def _s(v):
    return "—" if v is None else f"{v:.2f}s"


if __name__ == "__main__":
    main()
