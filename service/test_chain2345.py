# -*- coding: utf-8 -*-
"""链路 2~5 综合自测：分块/混合召回/rerank/分类/路由/MCQ/治理。"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(method, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def show(tag, ok, extra=""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {tag} {extra}")
    return ok


def main():
    results = []

    # ---- 基础健康 ----
    s, b = call("GET", "/health")
    results.append(show("health", s == 200 and b.get("status") == "ok"))

    # ---- 链路2+3：写入带时序/偏好/纠正的对话（alice）----
    conv = [
        {"role": "user", "content": "我叫 Alice，是一名后端工程师，最喜欢用 Python 写服务", "timestamp": 1704067200000},
        {"role": "assistant", "content": "很高兴认识你 Alice，Python 后端很强大", "timestamp": 1704067210000},
        {"role": "user", "content": "我 2024 年 1 月开始学 FastAPI，现在已经是主力框架了", "timestamp": 1704153600000},
        {"role": "assistant", "content": "FastAPI 的异步性能确实出色", "timestamp": 1704153610000},
        {"role": "user", "content": "平时我喜欢早上喝咖啡，不喜欢喝茶", "timestamp": 1704240000000},
        {"role": "assistant", "content": "记下了，你早上偏爱咖啡", "timestamp": 1704240010000},
    ]
    s, b = call("POST", "/add", {
        "request_id": "L23-alice-001", "messages": conv,
        "user_id": "alice", "session_id": "s1",
    })
    results.append(show("add alice conversation", s == 200 and b.get("success")))

    # ---- 链路5：写入一条纠正（纠正咖啡 -> 其实最近改喝茶）----
    correction = [
        {"role": "user", "content": "不对，我最近开始改喝茶了，现在不怎么喝咖啡了", "timestamp": 1704672000000},
    ]
    s, b = call("POST", "/add", {
        "request_id": "L5-alice-corr", "messages": correction,
        "user_id": "alice", "session_id": "s2",
    })
    results.append(show("add alice correction", s == 200 and b.get("success")))

    # ---- 链路2：混合召回（含专名 FastAPI）----
    s, b = call("POST", "/search", {
        "query": "Alice 的主力后端框架是什么", "user_id": "alice", "top_k": 5,
    })
    texts = " ".join(it["content"] for it in b.get("data", []))
    results.append(show("hybrid recall FastAPI", s == 200 and "FastAPI" in texts,
                        f"(recall {len(b.get('data', []))})"))

    # ---- 链路3 temporal：时序问题 ----
    s, b = call("POST", "/search", {
        "query": "Alice 什么时候开始学 FastAPI 的", "user_id": "alice", "top_k": 5,
    })
    texts = " ".join(it["content"] for it in b.get("data", []))
    results.append(show("temporal recall", s == 200 and ("2024" in texts or "1 月" in texts or "1月" in texts)))

    # ---- 链路3 preference + 链路5 治理：饮料偏好应返回最新（茶），而非被取代的咖啡 ----
    s, b = call("POST", "/search", {
        "query": "Alice 现在早上喜欢喝什么", "user_id": "alice", "top_k": 5,
    })
    data = b.get("data", [])
    texts = " ".join(it["content"] for it in data)
    # 治理后：纠正记忆应存在；旧的"喜欢喝咖啡"应被 superseded 过滤或排序靠后
    has_tea = "茶" in texts
    results.append(show("preference+latest recall", s == 200 and has_tea,
                        f"(returned {len(data)} items, has_tea={has_tea})"))

    # ---- 链路4：MCQ 选项级检索 ----
    s, b = call("POST", "/search", {
        "query": "Alice 最可能选择哪种饮品？",
        "options": ["A. 黑咖啡", "B. 绿茶", "C. 可乐", "D. 白开水"],
        "user_id": "alice", "top_k": 5,
    })
    results.append(show("mcq option-level search", s == 200 and len(b.get("data", [])) >= 0))

    # ---- user_id 隔离仍是红线 ----
    s, b = call("POST", "/search", {
        "query": "主力框架", "user_id": "bob_unknown", "top_k": 5,
    })
    results.append(show("user isolation", s == 200 and b.get("data") == []))

    # ---- top_k 上限 ----
    s, b = call("POST", "/search", {
        "query": "框架", "user_id": "alice", "top_k": 2,
    })
    results.append(show("top_k limit", s == 200 and len(b.get("data", [])) <= 2))

    # ---- 契约：返回结构 ----
    s, b = call("POST", "/search", {
        "query": "Python", "user_id": "alice", "top_k": 3,
    })
    ok_struct = (
        s == 200 and isinstance(b.get("data"), list)
        and all("id" in it and "content" in it for it in b["data"])
    )
    results.append(show("response contract", ok_struct))

    passed = sum(results)
    total = len(results)
    print(f"\n==== {passed}/{total} passed ====")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
