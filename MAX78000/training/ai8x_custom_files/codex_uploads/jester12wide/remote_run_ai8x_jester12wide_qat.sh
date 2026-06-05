#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT="/path/to/20bn-jester"
DATASET_ROOT="/path/to/20bn-jester-v1-complete"
AI8X="$PROJECT/ai8x-training"
CACHE="/tmp/jester_cache_12f64_full"
ENV_DIR="$PROJECT/envs/ai8x-py311"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/tmp/jester_ai8x_runs"
BACKUP_ROOT="$PROJECT/runs/ai8x_jester12wide_${RUN_ID}"
FP_NAME="ai8x_jester12wide_fp_${RUN_ID}"
QAT_NAME="ai8x_jester12wide_qat_${RUN_ID}"
LOG="$OUT_DIR/${QAT_NAME}.log"
BASELINE_DIR="$PROJECT/baselines/jesternet8_qat_20260520"

mkdir -p "$OUT_DIR" "$BACKUP_ROOT" "$BASELINE_DIR"

BASELINE_CKPT="$PROJECT/runs/ai8x_jesternet_qat_20260520_005821/artifacts/ai8x_jesternet_qat_20260520_005821_qat_best.pth.tar"
if [ -f "$BASELINE_CKPT" ]; then
  cp -n "$BASELINE_CKPT" "$BASELINE_DIR/"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

echo "[$(date)] starting 12-frame wider JesterNet run" | tee "$LOG"
echo "project=$PROJECT" | tee -a "$LOG"
echo "cache=$CACHE" | tee -a "$LOG"

if [ ! -f "$CACHE/train.pt" ] || [ ! -f "$CACHE/val.pt" ]; then
  echo "[$(date)] building full 12-frame 64x64 cache" | tee -a "$LOG"
  python "$PROJECT/train_jester_cached.py" cache \
    --dataset-root "$DATASET_ROOT" \
    --annotations-dir "$PROJECT/annotations" \
    --cache-dir "$CACHE" \
    --frames 12 \
    --image-size 64 \
    --train-samples 0 \
    --val-samples 0 \
    --seed 42 \
    2>&1 | tee -a "$LOG"
else
  echo "[$(date)] reusing existing cache" | tee -a "$LOG"
fi

cd "$AI8X"

echo "[$(date)] model summary" | tee -a "$LOG"
python train.py \
  --summary model \
  --model jester12wide \
  --dataset jester12 \
  --data "$CACHE" \
  --device MAX78000 \
  --use-bias \
  --gpus 0 \
  2>&1 | tee -a "$LOG"

echo "[$(date)] floating-point pretrain" | tee -a "$LOG"
python train.py \
  --lr 0.0005 \
  --optimizer adam \
  --epochs 60 \
  --batch-size 32 \
  --use-bias \
  --deterministic \
  --model jester12wide \
  --dataset jester12 \
  --data "$CACHE" \
  --validation-split 0 \
  --device MAX78000 \
  --gpus 0 \
  --workers 0 \
  --print-freq 200 \
  --name "$FP_NAME" \
  --out-dir "$OUT_DIR" \
  2>&1 | tee -a "$LOG"

FP_DIR="$(find "$OUT_DIR" -maxdepth 1 -type d -name "${FP_NAME}*" | sort | tail -1)"
FP_BEST="$(find "$FP_DIR" -type f -name '*best*.pth.tar' | sort | tail -1)"
if [ -z "$FP_BEST" ]; then
  echo "[$(date)] ERROR: no FP best checkpoint found in $FP_DIR" | tee -a "$LOG"
  exit 1
fi
echo "[$(date)] fp_best=$FP_BEST" | tee -a "$LOG"

echo "[$(date)] QAT fine-tune" | tee -a "$LOG"
python train.py \
  --lr 0.00005 \
  --optimizer adam \
  --epochs 40 \
  --batch-size 32 \
  --use-bias \
  --deterministic \
  --compress policies/schedule.yaml \
  --qat-policy policies/qat_policy_jester12wide.yaml \
  --model jester12wide \
  --dataset jester12 \
  --data "$CACHE" \
  --validation-split 0 \
  --device MAX78000 \
  --gpus 0 \
  --workers 0 \
  --print-freq 200 \
  --name "$QAT_NAME" \
  --out-dir "$OUT_DIR" \
  --exp-load-weights-from "$FP_BEST" \
  2>&1 | tee -a "$LOG"

echo "[$(date)] backing up selected artifacts" | tee -a "$LOG"
FP_BACKUP="$BACKUP_ROOT/fp_artifacts"
QAT_BACKUP="$BACKUP_ROOT/qat_artifacts"
QAT_DIR="$(find "$OUT_DIR" -maxdepth 1 -type d -name "${QAT_NAME}*" | sort | tail -1)"
cp -a "$FP_DIR" "$FP_BACKUP"
cp -a "$QAT_DIR" "$QAT_BACKUP"
cp "$LOG" "$BACKUP_ROOT/train.log"
cp "$CACHE/cache_config.json" "$BACKUP_ROOT/cache_config.json"
cp "$CACHE/labels.json" "$BACKUP_ROOT/labels.json"
find "$BACKUP_ROOT" -maxdepth 3 -type f \( -name '*best*.pth.tar' -o -name '*checkpoint*.pth.tar' -o -name '*.log' -o -name '*.json' \) -ls | tee -a "$LOG"
echo "[$(date)] 12-frame wider JesterNet complete backup_dir=$BACKUP_ROOT" | tee -a "$LOG"
