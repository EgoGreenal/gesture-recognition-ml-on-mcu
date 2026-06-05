#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
DATASET="/path/to/20bn-jester-v1-complete"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
CACHE_DIR="/tmp/jester_cache_8f64_5k_${RUN_ID}"
RUN_DIR="/tmp/jester_runs/cached_tinygesture_5k_${RUN_ID}"
LOG="/tmp/jester_runs/cached_tinygesture_5k_${RUN_ID}.log"

mkdir -p /tmp/jester_runs "$PROJECT/logs"
echo "$LOG" > "$PROJECT/logs/cached_train_latest_log.txt"
echo "$CACHE_DIR" > "$PROJECT/logs/cached_train_latest_cache.txt"
echo "$RUN_DIR" > "$PROJECT/logs/cached_train_latest_run.txt"

{
  echo "[$(date)] cached 5k trial start"
  echo "cache_dir=$CACHE_DIR"
  echo "run_dir=$RUN_DIR"
  echo "log=$LOG"
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader

  conda run --no-capture-output -n ai4eda python -u /tmp/train_jester_cached.py cache \
    --dataset-root "$DATASET" \
    --annotations-dir "$PROJECT/annotations" \
    --cache-dir "$CACHE_DIR" \
    --frames 8 \
    --image-size 64 \
    --train-samples 5000 \
    --val-samples 1000 \
    --seed 7

  conda run --no-capture-output -n ai4eda python -u /tmp/train_jester_cached.py train \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$RUN_DIR" \
    --frames 8 \
    --batch-size 128 \
    --epochs 5 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --seed 7

  echo "[$(date)] cached 5k trial complete"
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
} 2>&1 | tee "$LOG"
