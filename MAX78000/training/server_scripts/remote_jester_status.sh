#!/usr/bin/env bash
set -u

PROJECT="/path/to/20bn-jester"
DATASET="/path/to/20bn-jester-v1-complete"

echo "== processes =="
ps -fu "$USER" | grep -E "remote_prepare|jester|ai8x|git|curl|python" | grep -v grep || true

echo "== tmux =="
tmux ls 2>/dev/null || true

echo "== project tree =="
find "$PROJECT" -maxdepth 2 -mindepth 1 -printf "%M %s %p\n" | sort | head -120

echo "== ai8x status =="
if [[ -d "$PROJECT/ai8x-training/.git" ]]; then
  git -C "$PROJECT/ai8x-training" status --short
  git -C "$PROJECT/ai8x-training" rev-parse --short HEAD
else
  echo "ai8x-training missing or incomplete"
fi

echo "== annotations =="
find "$PROJECT/annotations" -maxdepth 2 -type f -printf "%s %p\n" 2>/dev/null | sort | head -80 || true

echo "== dataset frame count =="
find "$DATASET/20bn-jester-v1" -maxdepth 1 -mindepth 1 -type d | wc -l
