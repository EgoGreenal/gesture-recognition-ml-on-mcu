#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
SESSION="ai8x_jester_qat"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already running: $SESSION"
else
  tmux new-session -d -s "$SESSION" "bash $PROJECT/remote_run_ai8x_jesternet_qat.sh"
  echo "started tmux session: $SESSION"
fi

tmux ls
