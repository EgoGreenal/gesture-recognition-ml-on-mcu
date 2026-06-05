#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
SESSION="jester_train"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash '$PROJECT/remote_train_background.sh'"
tmux ls
echo "latest_log=$PROJECT/logs/train_latest.log"
