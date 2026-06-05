#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
DATASET="/path/to/20bn-jester-v1-complete"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
CACHE_DIR="/tmp/jester_cache_8f64_full_${RUN_ID}"
RUN_DIR="/tmp/jester_runs/cached_tinygesture_full_baseline_${RUN_ID}"
LOG="/tmp/jester_runs/cached_tinygesture_full_baseline_${RUN_ID}.log"
BACKUP_DIR="$PROJECT/runs/cached_tinygesture_full_baseline_${RUN_ID}"

mkdir -p /tmp/jester_runs "$PROJECT/logs"
echo "$LOG" > "$PROJECT/logs/cached_full_latest_log.txt"
echo "$CACHE_DIR" > "$PROJECT/logs/cached_full_latest_cache.txt"
echo "$RUN_DIR" > "$PROJECT/logs/cached_full_latest_run_tmp.txt"
echo "$BACKUP_DIR" > "$PROJECT/logs/cached_full_latest_backup.txt"

{
  echo "[$(date)] cached full baseline start"
  echo "cache_dir=$CACHE_DIR"
  echo "run_dir=$RUN_DIR"
  echo "backup_dir=$BACKUP_DIR"
  echo "log=$LOG"
  df -h /tmp
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader

  conda run --no-capture-output -n ai4eda python -u /tmp/train_jester_cached.py cache \
    --dataset-root "$DATASET" \
    --annotations-dir "$PROJECT/annotations" \
    --cache-dir "$CACHE_DIR" \
    --frames 8 \
    --image-size 64 \
    --train-samples 0 \
    --val-samples 0 \
    --seed 19

  conda run --no-capture-output -n ai4eda python -u /tmp/train_jester_cached.py train \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$RUN_DIR" \
    --frames 8 \
    --batch-size 128 \
    --epochs 5 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --seed 19

  echo "[$(date)] backing up selected artifacts"
  mkdir -p "$BACKUP_DIR"
  cp "$RUN_DIR/best.pth" "$BACKUP_DIR/"
  cp "$RUN_DIR/last.pth" "$BACKUP_DIR/"
  cp "$RUN_DIR/history.json" "$BACKUP_DIR/"
  cp "$CACHE_DIR/labels.json" "$BACKUP_DIR/"
  cp "$CACHE_DIR/cache_config.json" "$BACKUP_DIR/"
  cp "$LOG" "$BACKUP_DIR/train.log"
  ls -lh "$BACKUP_DIR"

  echo "[$(date)] cached full baseline complete"
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
} 2>&1 | tee "$LOG"
