#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
DATASET="/path/to/20bn-jester-v1-complete"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PROJECT/runs/tinygesture_smoke_${RUN_ID}"
LOG="/tmp/train_tinygesture_smoke_${RUN_ID}.log"

mkdir -p "$PROJECT/runs" "$PROJECT/logs"
echo "$LOG" > "$PROJECT/logs/train_latest_tmp_path.txt"
echo "$RUN_DIR" > "$PROJECT/logs/train_latest_run_dir.txt"

{
  echo "[$(date)] starting smoke training"
  echo "run_dir=$RUN_DIR"
  echo "log=$LOG"
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
  conda run --no-capture-output -n ai4eda python -u "$PROJECT/train_jester_multiframe.py" \
    --dataset-root "$DATASET" \
    --annotations-dir "$PROJECT/annotations" \
    --output-dir "$RUN_DIR" \
    --frames 8 \
    --image-size 64 \
    --batch-size 32 \
    --epochs 1 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --num-workers 0 \
    --max-train-samples 512 \
    --max-val-samples 128 \
    --seed 7
  echo "[$(date)] smoke training complete"
  cp "$LOG" "$PROJECT/logs/$(basename "$LOG")" || true
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
} 2>&1 | tee "$LOG"
