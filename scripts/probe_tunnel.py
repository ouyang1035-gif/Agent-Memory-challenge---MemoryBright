#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""隧道超时探测脚本（在本机运行，只用标准库，无需 pip 安装）。

用途：从你的电脑直接请求 DevBridge 隧道地址，看「请求超过 60s」时网关到底返回什么。

地址：
  慢服务（8010，我这边已起好，只有 sleep，不会碰线上检索）：
      https://errfowt2-8010.devbridge-s2.hwtunnel.com
  线上检索（8000，会产生真实 LLM 调用，评测期间别乱打）：
      https://errfowt2-8000.devbridge-s2.hwtunnel.com

用法示例：
  python probe_tunnel.py ping                              # 隧道通不通，2 秒内出结果
  python probe_tunnel.py slow --sec 30                     # 30s：应该 200
  python probe_tunnel.py slow --sec 75                     # 75s：看是不是 504
  python probe_tunnel.py stream --sec 75 --chunk 5         # 边算边吐字节，测「空闲超时」还是「总超时」
  python probe_tunnel.py sweep                             # 自动跑 30/55/60/65/70/90 秒，下结论
  python probe_tunnel.py search --user u_xxx --query "..." # 打线上 /search
"""

import argparse
import http.client
import json
import socket
import ssl
import sys
import time
from urllib.parse import urlparse

SLOW_BASE = "https://errfowt2-8010.devbridge-s2.hwtunnel.com"
SEARCH_BASE = "https://errfowt2-8000.devbridge-s2.hwtunnel.com"

# 默认客户端等 200s：故意比网关的 ~60s 长，否则我们自己的超时会掩盖网关的行为
DEFAULT_CLIENT_TIMEOUT = 200


def request(url, method="GET", body=None, headers=None, client_timeout=DEFAULT_CLIENT_TIMEOUT):
    """发一次请求，返回结果字典（不抛异常，异常也记成结果）。"""
    p = urlparse(url)
    if not p.hostname:
        raise SystemExit(f"地址不合法: {url}")
    port = p.port or (443 if p.scheme == "https" else 80)
    path = p.path or "/"
    if p.query:
        path += "?" + p.query

    hdrs = {"User-Agent": "probe-tunnel/1.0", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    if body is not None and "Content-Type" not in hdrs:
        hdrs["Content-Type"] = "application/json"

    if p.scheme == "https":
        conn = http.client.HTTPSConnection(
            p.hostname, port, timeout=client_timeout,
            context=ssl.create_default_context(),
        )
    else:
        conn = http.client.HTTPConnection(p.hostname, port, timeout=client_timeout)

    out = {
        "url": url, "status": None, "reason": "", "ttfb": None,
        "total": None, "headers": {}, "body": "", "error": "", "chunks": 0,
    }
    t0 = time.time()
    try:
        conn.request(method, path, body=body, headers=hdrs)
        resp = conn.getresponse()
        out["status"] = resp.status
        out["reason"] = resp.reason
        out["ttfb"] = time.time() - t0
        out["headers"] = dict(resp.getheaders())
        try:
            data = resp.read()
            out["body"] = data.decode("utf-8", "replace")
        except Exception as exc:  # 读 body 中途被断
            out["error"] = f"{type(exc).__name__}: {exc}"
        out["total"] = time.time() - t0
    except socket.timeout:
        out["total"] = time.time() - t0
        out["error"] = f"客户端等待超时（>{client_timeout}s，网关一直没回）"
    except Exception as exc:  # 连接被重置等
        out["total"] = time.time() - t0
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return out


def stream_request(url, client_timeout=DEFAULT_CLIENT_TIMEOUT):
    """逐块读响应，记录每块到达时间（测：只要一直有数据，网关还会不会断）。"""
    p = urlparse(url)
    port = p.port or (443 if p.scheme == "https" else 80)
    path = p.path + ("?" + p.query if p.query else "")
    conn = http.client.HTTPSConnection(
        p.hostname, port, timeout=client_timeout, context=ssl.create_default_context()
    )
    out = {"status": None, "chunks": [], "error": "", "ttfb": None}
    t0 = time.time()
    try:
        conn.request("GET", path, headers={"User-Agent": "probe-tunnel/1.0"})
        resp = conn.getresponse()
        out["status"] = resp.status
        out["ttfb"] = time.time() - t0
        while True:
            line = resp.read1(4096) if hasattr(resp, "read1") else resp.read(1)
            if not line:
                break
            out["chunks"].append((round(time.time() - t0, 1), line[:60].decode("utf-8", "replace").strip()))
    except socket.timeout:
        out["error"] = f"空闲超时：{client_timeout}s 内没等到新数据"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            conn.close()
        except Exception:
            pass
    out["total"] = time.time() - t0
    return out


def verdict(r):
    """把结果翻译成人话。"""
    body = (r.get("body") or "").lower()
    if r.get("status") == 504:
        return "504 —— 网关等上游超时（平台侧看到的就是这个）"
    if r.get("status") == 502:
        return "502 —— 网关连不上上游（隧道/服务没起来，跟超时不是一回事）"
    if r.get("status") == 200:
        return "200 —— 正常拿到数据"
    if "tunnel" in body and "timeout" in body:
        return "返回体里带 tunnel timeout —— 网关上游超时"
    if r.get("error"):
        return "没拿到响应：" + r["error"]
    return f"status={r.get('status')}"


def show(r, label=""):
    print("=" * 70)
    if label:
        print(label)
    print(f"URL      : {r['url']}")
    print(f"耗时     : 首字节={_f(r.get('ttfb'))}  总耗时={_f(r.get('total'))}")
    print(f"HTTP     : {r.get('status')} {r.get('reason') or ''}")
    print(f"结论     : {verdict(r)}")
    if r.get("headers"):
        for k in ("server", "date", "content-type", "content-length", "x-request-id"):
            if k in r["headers"]:
                print(f"  {k}: {r['headers'][k]}")
    if r.get("body"):
        b = r["body"]
        print(f"响应体   : {b[:500]}{'...' if len(b) > 500 else ''}")
    if r.get("error") and not r.get("status"):
        print(f"异常     : {r['error']}")
    print()


def _f(v):
    return "—" if v is None else f"{v:.2f}s"


def main():
    ap = argparse.ArgumentParser(description="DevBridge 隧道超时探测")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ping = sub.add_parser("ping", help="快速探活（/ping，立即返回）")
    p_ping.add_argument("--base", default=SLOW_BASE)

    p_slow = sub.add_parser("slow", help="让后端睡 N 秒再返回，看网关是否提前断")
    p_slow.add_argument("--sec", type=float, default=75)
    p_slow.add_argument("--base", default=SLOW_BASE)

    p_stream = sub.add_parser("stream", help="边算边吐字节，区分空闲超时与总超时")
    p_stream.add_argument("--sec", type=float, default=75)
    p_stream.add_argument("--chunk", type=float, default=5)
    p_stream.add_argument("--base", default=SLOW_BASE)

    p_sweep = sub.add_parser("sweep", help="自动扫多个时长，定位真实阈值")
    p_sweep.add_argument("--secs", default="30,55,60,65,70,90")
    p_sweep.add_argument("--base", default=SLOW_BASE)

    p_search = sub.add_parser("search", help="打线上 /search（会消耗 LLM，评测期间慎用）")
    p_search.add_argument("--user", required=True)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=100)
    p_search.add_argument("--base", default=SEARCH_BASE)

    ap.add_argument("--timeout", type=float, default=DEFAULT_CLIENT_TIMEOUT,
                    help="客户端最多等多久（默认 200s，要大于网关超时才能看到 504）")
    args = ap.parse_args()

    if args.cmd == "ping":
        r = request(args.base + "/ping", client_timeout=args.timeout)
        show(r, "探活")

    elif args.cmd == "slow":
        url = f"{args.base}/sleep?sec={args.sec:g}"
        print(f"发送：让后端睡 {args.sec:g}s，客户端最多等 {args.timeout:g}s ...\n")
        r = request(url, client_timeout=args.timeout)
        show(r, f"sleep {args.sec:g}s")

    elif args.cmd == "stream":
        url = f"{args.base}/sleep?sec={args.sec:g}&stream=1&chunk={args.chunk:g}"
        print(f"发送：流式，共 {args.sec:g}s，每 {args.chunk:g}s 一个包 ...\n")
        r = stream_request(url, client_timeout=args.timeout)
        print("=" * 70)
        print(f"HTTP     : {r['status']}  首字节={_f(r.get('ttfb'))}  总耗时={_f(r.get('total'))}")
        print(f"收到包数 : {len(r['chunks'])}")
        for t, s in r["chunks"]:
            print(f"  t={t:6.1f}s  {s}")
        if r.get("error"):
            print(f"异常     : {r['error']}")
        print()
        print("判读：如果一直收到包但最后仍被断 → 是总超时；"
              "如果长时间没包才断 → 是空闲超时。")

    elif args.cmd == "sweep":
        secs = [float(x) for x in args.secs.split(",") if x.strip()]
        rows = []
        for s in secs:
            print(f"\n>>> 测试 sleep={s:g}s ...", flush=True)
            r = request(f"{args.base}/sleep?sec={s:g}", client_timeout=args.timeout)
            rows.append((s, r.get("status"), r.get("total"), verdict(r)))
        print("\n" + "=" * 70)
        print(f"{'sleep(s)':>9} {'HTTP':>6} {'总耗时(s)':>10}  结论")
        for s, st, tot, v in rows:
            print(f"{s:>9g} {str(st):>6} {('%.1f' % tot) if tot else '-':>10}  {v}")

    elif args.cmd == "search":
        payload = json.dumps(
            {"user_id": args.user, "query": args.query, "top_k": args.top_k},
            ensure_ascii=False,
        ).encode()
        print(f"发送 /search，客户端最多等 {args.timeout:g}s ...\n")
        r = request(args.base + "/search", method="POST", body=payload,
                    client_timeout=args.timeout)
        show(r, f"/search top_k={args.top_k}")
        if r.get("status") == 200:
            try:
                data = json.loads(r["body"]).get("data", [])
                print(f"返回条数 : {len(data)}")
                for i, it in enumerate(data[:3], 1):
                    print(f"  top{i}: {str(it.get('content'))[:120]}  score={it.get('score')}")
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
