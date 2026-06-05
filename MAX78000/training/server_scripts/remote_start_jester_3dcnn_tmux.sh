#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/20bn-jester"
SESSION="jester_3dcnn"
LOG_DIR="$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/jester_3dcnn_${STAMP}.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session_exists=$SESSION"
  tmux ls | grep "$SESSION" || true
  exit 0
fi

tmux new-session -d -s "$SESSION" bash -lc \
  "cd '$ROOT' && bash remote_run_jester_3dcnn_8_12.sh 2>&1 | tee '$LOG_FILE'"
echo "$LOG_FILE" > "$LOG_DIR/jester_3dcnn_latest_log.txt"
echo "session=$SESSION"
echo "log=$LOG_FILE"
echo "run_dir_file=$LOG_DIR/jester_3dcnn_latest_run_dir.txt"
