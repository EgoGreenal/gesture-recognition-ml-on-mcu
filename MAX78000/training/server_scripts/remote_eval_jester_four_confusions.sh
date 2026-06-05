#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT="/path/to/20bn-jester"
AI8X="$PROJECT/ai8x-training"
ENV_DIR="$PROJECT/envs/ai8x-py311"
OUT="$PROJECT/runs/confusion_compare_20260520"

mkdir -p "$OUT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

cd "$AI8X"

run_eval() {
  local name="$1"
  local model="$2"
  local dataset="$3"
  local cache="$4"
  local policy="$5"
  local ckpt="$6"
  local log="$OUT/${name}.log"

  echo "[$(date)] evaluating $name" | tee "$log"
  python train.py \
    --evaluate \
    --confusion \
    --batch-size 64 \
    --use-bias \
    --deterministic \
    --qat-policy "$policy" \
    --model "$model" \
    --dataset "$dataset" \
    --data "$cache" \
    --validation-split 0 \
    --device MAX78000 \
    --gpus 0 \
    --workers 0 \
    --print-freq 200 \
    --name "confusion_${name}" \
    --out-dir "$OUT/logs" \
    --resume-from "$ckpt" \
    2>&1 | tee -a "$log"
}

run_eval \
  "motion_8p_wide" \
  "jestermotion8" \
  "jester_motion8" \
  "/tmp/jester_cache_motion8_64_full" \
  "policies/qat_policy_jestermotion8.yaml" \
  "$PROJECT/runs/ai8x_jestermotion8_20260520_211203/qat_artifacts/ai8x_jestermotion8_qat_20260520_211203_qat_best.pth.tar"

python "$PROJECT/parse_jester_confusions.py" \
  --out "$OUT" \
  --labels /tmp/jester_cache_8f64_full_20260519_192917/labels.json \
  --log baseline_8f="$OUT/baseline_8f.log" \
  --log raw_12f_wide="$OUT/raw_12f_wide.log" \
  --log motion_12p_wide="$OUT/motion_12p_wide.log" \
  --log motion_8p_wide="$OUT/motion_8p_wide.log"

echo "[$(date)] four-model confusion comparison complete: $OUT"
