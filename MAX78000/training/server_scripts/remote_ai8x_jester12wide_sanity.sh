#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT="/path/to/20bn-jester"
AI8X="$PROJECT/ai8x-training"
CACHE="/tmp/jester_cache_12f64_full"
ENV_DIR="$PROJECT/envs/ai8x-py311"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

cd "$AI8X"

python train.py \
  --model jester12wide \
  --dataset jester12 \
  --data "$CACHE" \
  --device MAX78000 \
  --use-bias \
  --summary model \
  --qat-policy policies/qat_policy_jester12wide.yaml
