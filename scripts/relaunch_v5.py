import json
import os
import subprocess

env = json.load(open("/tmp/v5_env_snapshot.json"))
env.update({
    "RERANK_DEPTH": "20",
    "RERANK_BATCH_SIZE": "20",
    "RERANK_WINDOW_STRIDE": "0",
    "RERANK_CANDIDATE_LIMIT": "24",
    "RERANK_TOTAL_TIMEOUT_SECONDS": "45",
    "RERANK_MAX_CHARS": "1000",
})
log = open("/tmp/v5_8000_restart.log", "ab")
p = subprocess.Popen(
    ["python3", "-u", "main.py"],
    cwd="/workspace/amc_eval/v5",
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("launched pid", p.pid)
