#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
SESSION="ai8x_jestermotion12_qat_lowlr"

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "tmux session already exists: $SESSION"
  tmux ls
  exit 0
}

tmux new-session -d -s "$SESSION" "bash '$PROJECT/remote_run_ai8x_jestermotion12_qat_lowlr.sh'"
echo "started tmux session: $SESSION"
tmux ls
