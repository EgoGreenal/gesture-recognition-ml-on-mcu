#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT="/path/to/20bn-jester"
AI8X="$PROJECT/ai8x-training"
CACHE="/tmp/jester_cache_motion12_64_full"
ENV_DIR="$PROJECT/envs/ai8x-py311"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$PROJECT/runtime/ai8x_runs"
SCRATCH_TMP="$PROJECT/tmp"
BACKUP_ROOT="$PROJECT/runs/ai8x_jestermotion12_qat_lowlr_${RUN_ID}"
QAT_NAME="ai8x_jestermotion12_qat_lowlr_${RUN_ID}"
LOG="$OUT_DIR/${QAT_NAME}.log"
FP_BEST="$PROJECT/runs/ai8x_jestermotion12_20260520_184836/fp_artifacts/ai8x_jestermotion12_fp_20260520_184836_best.pth.tar"

mkdir -p "$OUT_DIR" "$BACKUP_ROOT" "$SCRATCH_TMP"
export TMPDIR="$SCRATCH_TMP"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

echo "[$(date)] starting motion12 improved-QAT experiment" | tee "$LOG"
echo "project=$PROJECT" | tee -a "$LOG"
echo "cache=$CACHE" | tee -a "$LOG"
echo "fp_best=$FP_BEST" | tee -a "$LOG"
echo "qat_lr=0.00002 qat_epochs=20" | tee -a "$LOG"

if [ ! -f "$CACHE/train.pt" ] || [ ! -f "$CACHE/val.pt" ]; then
  echo "[$(date)] ERROR: missing motion12 cache at $CACHE" | tee -a "$LOG"
  exit 1
fi
if [ ! -f "$FP_BEST" ]; then
  echo "[$(date)] ERROR: missing FP best checkpoint $FP_BEST" | tee -a "$LOG"
  exit 1
fi

cd "$AI8X"

python train.py \
  --summary model \
  --model jestermotion12 \
  --dataset jester_motion12 \
  --data "$CACHE" \
  --device MAX78000 \
  --use-bias \
  --gpus 0 \
  2>&1 | tee -a "$LOG"

echo "[$(date)] low-LR short QAT fine-tune" | tee -a "$LOG"
python train.py \
  --lr 0.00002 \
  --optimizer adam \
  --epochs 20 \
  --batch-size 32 \
  --use-bias \
  --deterministic \
  --compress policies/schedule.yaml \
  --qat-policy policies/qat_policy_jestermotion12.yaml \
  --model jestermotion12 \
  --dataset jester_motion12 \
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
cp -a "$QAT_DIR" "$BACKUP_ROOT/qat_artifacts"
cp "$LOG" "$BACKUP_ROOT/train.log"
cp "$CACHE/cache_config.json" "$BACKUP_ROOT/cache_config.json"
cp "$CACHE/labels.json" "$BACKUP_ROOT/labels.json"
find "$BACKUP_ROOT" -maxdepth 3 -type f \( -name '*best*.pth.tar' -o -name '*checkpoint*.pth.tar' -o -name '*.log' -o -name '*.json' \) -ls | tee -a "$LOG"
echo "[$(date)] motion12 improved-QAT complete backup_dir=$BACKUP_ROOT" | tee -a "$LOG"
