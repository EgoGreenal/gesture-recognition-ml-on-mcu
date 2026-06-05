#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT="/path/to/20bn-jester"
DATASET_ROOT="/path/to/20bn-jester-v1-complete"
AI8X="$PROJECT/ai8x-training"
CACHE="$PROJECT/cache/jester_cache_signedmotion12_64_full"
ENV_DIR="$PROJECT/envs/ai8x-py311"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$PROJECT/runtime/ai8x_runs"
SCRATCH_TMP="$PROJECT/tmp"
BACKUP_ROOT="$PROJECT/runs/ai8x_jestersignedmotion12_${RUN_ID}"
FP_NAME="ai8x_jestersignedmotion12_fp_${RUN_ID}"
QAT_NAME="ai8x_jestersignedmotion12_qat_${RUN_ID}"
LOG="$OUT_DIR/${QAT_NAME}.log"
FP_EPOCHS="${FP_EPOCHS:-60}"
QAT_EPOCHS="${QAT_EPOCHS:-30}"

mkdir -p "$OUT_DIR" "$BACKUP_ROOT" "$SCRATCH_TMP"
export TMPDIR="$SCRATCH_TMP"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

echo "[$(date)] starting signed-motion JesterNet run" | tee "$LOG"
echo "project=$PROJECT" | tee -a "$LOG"
echo "cache=$CACHE" | tee -a "$LOG"
echo "encoding=signed_motion: plane0=center frame, plane1-11=(curr-prev)/2+128" | tee -a "$LOG"
echo "fp_epochs=$FP_EPOCHS qat_epochs=$QAT_EPOCHS" | tee -a "$LOG"

if [ ! -f "$CACHE/train.pt" ] || [ ! -f "$CACHE/val.pt" ]; then
  echo "[$(date)] building full signed-motion 12-plane 64x64 cache" | tee -a "$LOG"
  python "$PROJECT/train_jester_cached.py" cache \
    --dataset-root "$DATASET_ROOT" \
    --annotations-dir "$PROJECT/annotations" \
    --cache-dir "$CACHE" \
    --frames 12 \
    --image-size 64 \
    --encoding signed_motion \
    --train-samples 0 \
    --val-samples 0 \
    --seed 42 \
    2>&1 | tee -a "$LOG"
else
  echo "[$(date)] reusing existing signed-motion cache" | tee -a "$LOG"
fi

cd "$AI8X"

echo "[$(date)] model summary" | tee -a "$LOG"
python train.py \
  --summary model \
  --model jestersignedmotion12 \
  --dataset jester_signedmotion12 \
  --data "$CACHE" \
  --device MAX78000 \
  --use-bias \
  --gpus 0 \
  2>&1 | tee -a "$LOG"

echo "[$(date)] floating-point pretrain" | tee -a "$LOG"
python train.py \
  --lr 0.0005 \
  --optimizer adam \
  --epochs "$FP_EPOCHS" \
  --batch-size 32 \
  --use-bias \
  --deterministic \
  --model jestersignedmotion12 \
  --dataset jester_signedmotion12 \
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
  --epochs "$QAT_EPOCHS" \
  --batch-size 32 \
  --use-bias \
  --deterministic \
  --compress policies/schedule.yaml \
  --qat-policy policies/qat_policy_jestersignedmotion12.yaml \
  --model jestersignedmotion12 \
  --dataset jester_signedmotion12 \
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
QAT_DIR="$(find "$OUT_DIR" -maxdepth 1 -type d -name "${QAT_NAME}*" | sort | tail -1)"
cp -a "$FP_DIR" "$BACKUP_ROOT/fp_artifacts"
cp -a "$QAT_DIR" "$BACKUP_ROOT/qat_artifacts"
cp "$LOG" "$BACKUP_ROOT/train.log"
cp "$CACHE/cache_config.json" "$BACKUP_ROOT/cache_config.json"
cp "$CACHE/labels.json" "$BACKUP_ROOT/labels.json"
find "$BACKUP_ROOT" -maxdepth 3 -type f \( -name '*best*.pth.tar' -o -name '*checkpoint*.pth.tar' -o -name '*.log' -o -name '*.json' \) -ls | tee -a "$LOG"
echo "[$(date)] signed-motion JesterNet complete backup_dir=$BACKUP_ROOT" | tee -a "$LOG"
