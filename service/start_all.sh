#!/usr/bin/env bash
# 一键启动 / 重启本地评测全部服务（embed + 自研 + mem0 + tencentdb）。
#
# 用法：
#   bash start_all.sh            # 重启全部
#   bash start_all.sh status     # 只看状态
#   bash start_all.sh stop       # 只停止
#
# 说明：
#   - 使用 setsid 让进程完全脱离当前会话，避免终端/会话切换时被回收
#   - 用 pid 文件精确管理，避免 pkill 误杀
#   - embed 服务预加载模型，启动等待时间较长（首次约 20~40s），脚本会轮询 health
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/.run"
LOG_DIR="/tmp/amc_logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

EMBED_PORT=8080
SELF_PORT=8000
MEM0_PORT=8001
TCDB_PORT=8002

# 限制 OpenMP/BLAS 线程数：宿主机 640 核，默认会按核数开线程池；
# 叠加容器 cgroup pids 上限（10000）会在并发时触发
# "libgomp: Thread creation failed: Resource temporarily unavailable" 导致服务崩溃。
# 2026-10-28 修正：容器 cgroup 配额是 40 核（不是此前以为的 4 核）。
# 非 embed 服务保持 4 线程（它们只做 numpy 小矩阵运算，开多无收益）；
# embed 单独给 16 线程 × 2 worker（embed_server.py 内部会用该值覆盖 OMP/MKL）。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export EMBED_TORCH_THREADS="${EMBED_TORCH_THREADS:-16}"
export EMBED_WORKERS="${EMBED_WORKERS:-2}"
# 单条输入的 token 上限：bge-m3 默认 8192，长文本编码开销随 token 线性增长
export EMBED_MAX_SEQ_LEN="${EMBED_MAX_SEQ_LEN:-1024}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# 服务定义：name|port|工作目录|启动命令
SERVICES=(
  "embed|$EMBED_PORT|$ROOT|python3 -u embed_server.py"
  "tencentdb|$TCDB_PORT|$ROOT/tencentdb_service|SERVICE_PORT=8002 python3 -u main.py"
  "mem0|$MEM0_PORT|$ROOT/mem0_service|python3 -u main.py"
  "self|$SELF_PORT|$ROOT|python3 -u main.py"
)

pid_file() { echo "$RUN_DIR/$1.pid"; }

is_alive() {
  local pf; pf="$(pid_file "$1")"
  [ -f "$pf" ] || return 1
  local pid; pid="$(cat "$pf" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

stop_one() {
  local name="$1" pf; pf="$(pid_file "$name")"
  if [ -f "$pf" ]; then
    local pid; pid="$(cat "$pf" 2>/dev/null)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
      for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.3; done
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
      echo "  stopped $name (pid=$pid)"
    fi
    rm -f "$pf"
  fi
}

stop_all() {
  echo "== 停止全部服务 =="
  for entry in "${SERVICES[@]}"; do
    IFS='|' read -r name _ _ _ <<< "$entry"
    stop_one "$name"
  done
}

start_one() {
  local name="$1" port="$2" wd="$3" cmd="$4"
  local pf; pf="$(pid_file "$name")"
  if is_alive "$name"; then
    echo "  $name 已在运行 (pid=$(cat "$pf"))"
    return 0
  fi
  # 关键：setsid 可能 fork，$! 记录的不是真实服务进程。
  # 改为让目标进程自己把 $$ 写入 pidfile，exec 后同一 pid 即服务进程。
  setsid bash -c "cd '$wd' && echo \$\$ > '$pf' && exec env $cmd" \
    > "$LOG_DIR/$name.log" 2>&1 < /dev/null &
  # 等待目标进程写入 pidfile
  for _ in $(seq 1 50); do [ -s "$pf" ] && break; sleep 0.2; done
  echo "  started $name -> port $port (pid=$(cat "$pf" 2>/dev/null))"
}

wait_health() {
  local name="$1" port="$2" timeout="${3:-60}"
  for i in $(seq 1 "$timeout"); do
    if curl -s -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

status_all() {
  echo "== 服务状态 =="
  for entry in "${SERVICES[@]}"; do
    IFS='|' read -r name port _ _ <<< "$entry"
    local h; h="$(curl -s -m 3 "http://127.0.0.1:$port/health" 2>/dev/null)"
    if [ -n "$h" ]; then
      local pid=""; is_alive "$name" && pid="$(cat "$(pid_file "$name")")"
      printf "  %-10s :%-5s OK   pid=%s  %s\n" "$name" "$port" "$pid" "$h"
    else
      printf "  %-10s :%-5s DOWN\n" "$name" "$port"
    fi
  done
}

cmd="${1:-restart}"

case "$cmd" in
  status) status_all ;;
  stop)   stop_all ;;
  start)
    echo "== 启动全部服务 =="
    start_one embed "$EMBED_PORT" "$ROOT" "python3 -u embed_server.py"
    echo "  等待 embed 预加载模型..."
    if wait_health embed "$EMBED_PORT" 120; then echo "  embed 就绪"; else echo "  !! embed 启动超时，检查 $LOG_DIR/embed.log"; fi
    for entry in "${SERVICES[@]}"; do
      IFS='|' read -r name port wd c <<< "$entry"
      [ "$name" = "embed" ] && continue
      start_one "$name" "$port" "$wd" "$c"
    done
    sleep 3
    status_all
    ;;
  restart|*)
    stop_all
    sleep 1
    echo "== 启动全部服务 =="
    start_one embed "$EMBED_PORT" "$ROOT" "python3 -u embed_server.py"
    echo "  等待 embed 预加载模型..."
    if wait_health embed "$EMBED_PORT" 120; then echo "  embed 就绪"; else echo "  !! embed 启动超时，检查 $LOG_DIR/embed.log"; fi
    for entry in "${SERVICES[@]}"; do
      IFS='|' read -r name port wd c <<< "$entry"
      [ "$name" = "embed" ] && continue
      start_one "$name" "$port" "$wd" "$c"
    done
    sleep 3
    status_all
    ;;
esac
