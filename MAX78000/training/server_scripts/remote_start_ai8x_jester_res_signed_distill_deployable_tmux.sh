#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
SESSION="${SESSION:-jester_res_signed_distill}"
SCRIPT="$PROJECT/remote_run_ai8x_jester_res_signed_distill_deployable.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  tmux capture-pane -pt "$SESSION" | tail -80
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash '$SCRIPT'"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "status: tmux capture-pane -pt $SESSION | tail -80"
