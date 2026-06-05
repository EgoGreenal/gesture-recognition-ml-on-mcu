#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT="/path/to/20bn-jester"
AI8X="$PROJECT/ai8x-training"
CACHE="/tmp/jester_cache_8f64_full_20260519_192917"
ENV_DIR="$PROJECT/envs/ai8x-py311"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/tmp/jester_ai8x_runs"
BACKUP_DIR="$PROJECT/runs/ai8x_jesternet_qat_${RUN_ID}"
LOG="$OUT_DIR/ai8x_jesternet_qat_${RUN_ID}.log"

mkdir -p "$OUT_DIR" "$BACKUP_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

cd "$AI8X"

echo "[$(date)] starting ai8x JesterNet QAT" | tee "$LOG"
python train.py \
  --lr 0.00005 \
  --optimizer adam \
  --epochs 40 \
  --batch-size 64 \
  --use-bias \
  --deterministic \
  --compress policies/schedule.yaml \
  --qat-policy policies/qat_policy_jesternet.yaml \
  --model jesternet \
  --dataset jester \
  --data "$CACHE" \
  --validation-split 0 \
  --device MAX78000 \
  --gpus 0 \
  --workers 0 \
  --print-freq 200 \
  --name ai8x_jesternet_qat_${RUN_ID} \
  --out-dir "$OUT_DIR" \
  --exp-load-weights-from "$PROJECT/checkpoints/jesternet_ai8x.pth.tar" \
  2>&1 | tee -a "$LOG"

echo "[$(date)] backing up selected artifacts" | tee -a "$LOG"
LATEST="$(ls -dt "$OUT_DIR"/ai8x_jesternet_qat_${RUN_ID}*/ | head -1)"
cp -a "$LATEST" "$BACKUP_DIR/artifacts"
cp "$LOG" "$BACKUP_DIR/train.log"
find "$BACKUP_DIR" -maxdepth 3 -type f \( -name '*best*.pth.tar' -o -name '*checkpoint*.pth.tar' -o -name '*.log' \) -ls | tee -a "$LOG"
echo "[$(date)] ai8x JesterNet QAT complete backup_dir=$BACKUP_DIR" | tee -a "$LOG"
