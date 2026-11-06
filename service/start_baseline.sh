#!/usr/bin/env bash
# 启动 baseline_74_77（官方 smoke 74.77 那一版）
# 用法: bash start_baseline.sh [port] [store_dir]
set -euo pipefail

PORT="${1:-8000}"
STORE="${2:-/workspace/amc_eval/v4_smoke_store}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export SERVICE_PORT="$PORT"
export SQLITE_DB_PATH="$STORE/memory.db"
export QDRANT_LOCAL_PATH="$STORE/qdrant"
export API_IO_LOG_DIR="/tmp/amc_logs_baseline_${PORT}"
export ENABLE_SUMMARY_MODE=true
export SUMMARY_GATE_BY_INTENT=true
export ENABLE_SUMMARY_LAYER=true
export SUMMARY_MAX_CHARS=800

nohup python3 -u main.py > "/tmp/svc_${PORT}_baseline.log" 2>&1 &
echo "baseline started pid=$! port=$PORT store=$STORE"
