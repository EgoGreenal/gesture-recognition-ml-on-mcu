#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
DATASET="/path/to/20bn-jester-v1-complete"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PROJECT/runs/tinygesture_${RUN_ID}"
LOG="$PROJECT/logs/train_tinygesture_${RUN_ID}.log"
LATEST="$PROJECT/logs/train_latest.log"

mkdir -p "$PROJECT/runs" "$PROJECT/logs"
ln -sfn "$LOG" "$LATEST"

{
  echo "[$(date)] starting training"
  echo "run_dir=$RUN_DIR"
  echo "log=$LOG"
  echo "dataset=$DATASET"
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
  conda run --no-capture-output -n ai4eda python -u "$PROJECT/train_jester_multiframe.py" \
    --dataset-root "$DATASET" \
    --annotations-dir "$PROJECT/annotations" \
    --output-dir "$RUN_DIR" \
    --frames 8 \
    --image-size 64 \
    --batch-size 64 \
    --epochs 2 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --num-workers 4 \
    --max-train-samples 5000 \
    --max-val-samples 1000 \
    --seed 7
  echo "[$(date)] training complete"
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
} 2>&1 | tee "$LOG"
