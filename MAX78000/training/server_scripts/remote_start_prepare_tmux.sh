#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
SESSION="jester_prepare"

tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f "$PROJECT/remote_prepare_background.sh" 2>/dev/null || true
pkill -f "git clone https://github.com/analogdevicesinc/ai8x-training.git" 2>/dev/null || true
pkill -f "git-remote-https origin https://github.com/analogdevicesinc/ai8x-training.git" 2>/dev/null || true
pkill -f "git index-pack.*ai8x-training" 2>/dev/null || true
pkill -f "rm -rf ai8x-training" 2>/dev/null || true
sleep 2
tmux new-session -d -s "$SESSION" "bash '$PROJECT/remote_prepare_background.sh'"
tmux ls
echo "log=$PROJECT/logs/prepare_resources.log"
