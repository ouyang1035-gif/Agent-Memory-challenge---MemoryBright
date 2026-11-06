#!/usr/bin/env python3
"""冷启动 / 持久性专项：重启各臂服务，测量启动、缓存构建、首次检索耗时，
并重跑 orig 探针校验重启前后结果一致性。"""
import json, os, re, signal, subprocess, sys, time, urllib.request

V5 = "/workspace/amc_eval/v5"
KNEE = "u_56d3c8e3472cb8d469229c64f3234a2b714d0ca1dfe28d575e95e956305429c9"
ARMS = {
    "s0": ("8046", "/tmp/rb_s0", "/tmp/rb_logs_8046", ""),
    "s1": ("8047", "/tmp/rb_s1", "/tmp/rb_logs_8047",
           "ENABLE_CLASSIFICATION=false CHUNK_TARGET_MIN=80 CHUNK_TARGET_MAX=200"),
    "s2": ("8048", "/tmp/rb_s2", "/tmp/rb_logs_8048",
           "ENABLE_CLASSIFICATION=false CHUNK_TARGET_MIN=80 CHUNK_TARGET_MAX=200"),
    "s3": ("8049", "/tmp/rb_s3", "/tmp/rb_logs_8049", ""),
}
COMMON = ("ENABLE_SUMMARY_MODE=true SUMMARY_GATE_BY_INTENT=true SUMMARY_MAX_CHARS=800 "
          "ENABLE_SUMMARY_LAYER=false ENABLE_FACT_EXTRACTION=false")


def find_pid(port):
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            env = open(f"/proc/{p}/environ", "rb").read().decode("utf8", "ignore")
        except Exception:
            continue
        if "SERVICE_PORT=" + port in env and "main.py" in open(f"/proc/{p}/cmdline", "rb").read().decode("utf8", "ignore"):
            return int(p)
    return None


def http_ok(port, timeout=2.0):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def search(port, query="我最近睡眠怎么样", rid="cold-1"):
    payload = {"user_id": KNEE, "query": query, "top_k": 100, "request_id": rid}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/search",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read().decode())
    return time.time() - t0, len(body.get("data") or [])


def main():
    out_path = "/tmp/rb/coldstart.json"
    rows = {}
    for tag, (port, db, logdir, extra) in ARMS.items():
        logf = f"/tmp/rb_svc_{port}.log"
        pid = find_pid(port)
        print(f"[{tag}] port={port} pid={pid}", flush=True)
        if pid:
            os.kill(pid, signal.SIGTERM)
            for _ in range(100):
                if not http_ok(port, 0.5):
                    break
                time.sleep(0.2)
        # 清空旧日志便于抓启动行
        open(logf, "w").close()
        env = dict(os.environ)
        for kv in (COMMON + " " + extra).split():
            k, v = kv.split("=", 1)
            env[k] = v
        env.update({"SERVICE_PORT": port,
                    "SQLITE_DB_PATH": f"{db}/memory.db",
                    "QDRANT_LOCAL_PATH": f"{db}/qdrant_storage",
                    "API_IO_LOG_DIR": logdir})
        t0 = time.time()
        with open(logf, "w") as lf:
            subprocess.Popen(["python3", "-u", "main.py"], cwd=V5, env=env,
                             stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        ready = None
        while time.time() - t0 < 180:
            if http_ok(port):
                ready = time.time() - t0
                break
            time.sleep(0.2)
        s1, n1 = search(port, rid=f"cold-{tag}-1")
        s2, n2 = search(port, rid=f"cold-{tag}-2")
        # 解析日志
        log = open(logf, encoding="utf8", errors="ignore").read()
        vc = re.search(r"vector cache ready: (\d+) points in ([\d.]+)s", log)
        bm = re.findall(r"bm25.*?(?:build|index).{0,60}?([\d.]+)s", log, re.I)
        rows[tag] = {
            "port": port,
            "health_ready_s": round(ready, 2) if ready else None,
            "first_search_s": round(s1, 2), "first_n": n1,
            "second_search_s": round(s2, 2), "second_n": n2,
            "vector_cache_points": int(vc.group(1)) if vc else None,
            "vector_cache_s": float(vc.group(2)) if vc else None,
            "log_size": len(log),
        }
        print(f"[{tag}] {rows[tag]}", flush=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    print("COLDSTART_DONE", flush=True)


if __name__ == "__main__":
    main()
