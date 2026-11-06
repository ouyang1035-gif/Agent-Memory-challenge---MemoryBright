"""链路 1 端到端自测脚本：/health -> /add -> /search -> 幂等 -> 隔离性。"""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(method, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main():
    print("=== 1. /health ===")
    s, b = call("GET", "/health")
    print(s, b)
    assert s == 200 and b.get("status") == "ok"

    print("\n=== 2. /add (user-alice) ===")
    add_payload = {
        "request_id": "test-001",
        "messages": [
            {"role": "user", "content": "我喜欢用 Python 做后端开发", "timestamp": 1704067200000},
            {"role": "assistant", "content": "Python 确实很适合后端", "timestamp": 1704067210000},
            {"role": "user", "content": "我上周开始学习 FastAPI 框架", "timestamp": 1704153600000},
        ],
        "user_id": "user-alice",
        "session_id": "sess-1",
    }
    s, b = call("POST", "/add", add_payload)
    print(s, b)
    assert s == 200 and b["success"] is True and b["request_id"] == "test-001"

    print("\n=== 3. /add 幂等（同 request_id 重复）===")
    s, b = call("POST", "/add", add_payload)
    print(s, b, "(应成功且不重复写入)")
    assert s == 200 and b["success"] is True

    print("\n=== 4. /add 另一个用户 (user-bob) ===")
    bob_payload = {
        "request_id": "test-002",
        "messages": [
            {"role": "user", "content": "我是 Java 程序员，主要做安卓开发", "timestamp": 1704067200000},
        ],
        "user_id": "user-bob",
        "session_id": "sess-9",
    }
    s, b = call("POST", "/add", bob_payload)
    print(s, b)
    assert s == 200

    print("\n=== 5. /search 检索 alice 的偏好 ===")
    s, b = call("POST", "/search", {
        "query": "用户喜欢什么编程语言和框架？",
        "user_id": "user-alice",
        "top_k": 100,
    })
    print(s)
    for i, item in enumerate(b["data"]):
        print(f"  [{i}] score={item.get('score'):.4f} {item['content'][:30]}")
    assert s == 200 and isinstance(b["data"], list)
    assert len(b["data"]) <= 100, "不能超过 top_k"
    assert all("Java" not in it["content"] or "安卓" not in it["content"] for it in b["data"] if it["content"]), \
        "不应返回 bob 的记忆"
    assert len(b["data"]) > 0, "alice 应该有召回"

    print("\n=== 6. /search user_id 隔离：bob 查不到 alice ===")
    s, b = call("POST", "/search", {
        "query": "用户喜欢 Python 吗",
        "user_id": "user-bob",
        "top_k": 100,
    })
    contents = [it["content"] for it in b["data"]]
    print(s, f"bob 召回 {len(contents)} 条:", contents)
    assert all("Python" not in c and "FastAPI" not in c for c in contents), "bob 不应看到 alice 的记忆"

    print("\n=== 7. /search 不存在的用户返回空数组 ===")
    s, b = call("POST", "/search", {
        "query": "随便什么问题",
        "user_id": "user-nonexistent",
        "top_k": 100,
    })
    print(s, b)
    assert s == 200 and b["data"] == []

    print("\n=== 8. top_k 限制测试（top_k=1）===")
    s, b = call("POST", "/search", {
        "query": "编程",
        "user_id": "user-alice",
        "top_k": 1,
    })
    print(s, f"返回 {len(b['data'])} 条")
    assert len(b["data"]) <= 1, "必须 <= top_k"

    print("\n✅ 全部自测通过！")


if __name__ == "__main__":
    main()
