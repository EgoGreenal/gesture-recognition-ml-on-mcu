#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
SESSION="ai8x_jestermixed12"

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "tmux session already exists: $SESSION"
  tmux ls
  exit 0
}

tmux new-session -d -s "$SESSION" "bash '$PROJECT/remote_run_ai8x_jestermixed12_qat.sh'"
echo "started tmux session: $SESSION"
tmux ls
