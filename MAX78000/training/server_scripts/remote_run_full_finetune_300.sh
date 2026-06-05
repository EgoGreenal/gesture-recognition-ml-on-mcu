#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
CACHE_DIR="/tmp/jester_cache_8f64_full_20260519_192917"
RESUME="$PROJECT/runs/cached_tinygesture_full_baseline_20260519_192917/best.pth"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/tmp/jester_runs/cached_tinygesture_full_finetune300_${RUN_ID}"
LOG="/tmp/jester_runs/cached_tinygesture_full_finetune300_${RUN_ID}.log"
BACKUP_DIR="$PROJECT/runs/cached_tinygesture_full_finetune300_${RUN_ID}"

mkdir -p /tmp/jester_runs "$PROJECT/logs"
echo "$LOG" > "$PROJECT/logs/full_finetune300_latest_log.txt"
echo "$CACHE_DIR" > "$PROJECT/logs/full_finetune300_latest_cache.txt"
echo "$RUN_DIR" > "$PROJECT/logs/full_finetune300_latest_run_tmp.txt"
echo "$BACKUP_DIR" > "$PROJECT/logs/full_finetune300_latest_backup.txt"

{
  echo "[$(date)] full 300-epoch fine-tune start"
  echo "cache_dir=$CACHE_DIR"
  echo "resume=$RESUME"
  echo "run_dir=$RUN_DIR"
  echo "backup_dir=$BACKUP_DIR"
  echo "log=$LOG"
  df -h /tmp
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader

  conda run --no-capture-output -n ai4eda python -u /tmp/train_jester_cached.py train \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$RUN_DIR" \
    --frames 8 \
    --batch-size 128 \
    --epochs 300 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --seed 23 \
    --resume-checkpoint "$RESUME" \
    --resume-history \
    --continue-epoch-numbers

  echo "[$(date)] backing up selected artifacts"
  mkdir -p "$BACKUP_DIR"
  cp "$RUN_DIR/best.pth" "$BACKUP_DIR/"
  cp "$RUN_DIR/last.pth" "$BACKUP_DIR/"
  cp "$RUN_DIR/history.json" "$BACKUP_DIR/"
  cp "$CACHE_DIR/labels.json" "$BACKUP_DIR/"
  cp "$CACHE_DIR/cache_config.json" "$BACKUP_DIR/"
  cp "$LOG" "$BACKUP_DIR/train.log"
  ls -lh "$BACKUP_DIR"

  echo "[$(date)] full 300-epoch fine-tune complete"
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
} 2>&1 | tee "$LOG"
