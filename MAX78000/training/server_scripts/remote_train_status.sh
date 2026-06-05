#!/usr/bin/env bash
set -u

PROJECT="/path/to/20bn-jester"

echo "== tmux =="
tmux ls 2>/dev/null || true

echo "== training processes =="
ps -fu "$USER" | grep -E "train_jester|conda|python|tmux" | grep -v grep || true

echo "== gpu =="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader

echo "== latest log =="
wc -l "$PROJECT/logs/train_latest.log" 2>/dev/null || true
tail -120 "$PROJECT/logs/train_latest.log" 2>/dev/null || true

echo "== latest run files =="
latest_run=$(ls -dt "$PROJECT"/runs/tinygesture_* 2>/dev/null | head -1 || true)
echo "latest_run=$latest_run"
if [[ -n "$latest_run" ]]; then
  find "$latest_run" -maxdepth 1 -type f -printf "%s %p\n" | sort
  echo "config:"
  cat "$latest_run/config.json" 2>/dev/null || true
fi
