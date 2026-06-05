#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
SESSION="${SESSION:-ai8x_jester_huge_teacher_qat_sweep}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  tmux list-sessions | grep "$SESSION" || true
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$PROJECT/remote_run_ai8x_jester_huge_teacher_qat_sweep.sh'"
echo "started tmux session: $SESSION"
tmux list-sessions | grep "$SESSION"
