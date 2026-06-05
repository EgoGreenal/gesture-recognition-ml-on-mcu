#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT="/path/to/20bn-jester"
AI8X="$PROJECT/ai8x-training"
ENV_DIR="$PROJECT/envs/ai8x-py311"
CACHE="/tmp/jester_cache_motion12_64_full"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

cd "$AI8X"

python -m py_compile \
  "$PROJECT/train_jester_cached.py" \
  models/jestermotion12res.py \
  models/jestersignedmotion12.py \
  datasets/jester_signedmotion12.py \
  datasets/jester_motion12_hard.py

python train.py \
  --summary model \
  --model jestermotion12res \
  --dataset jester_motion12 \
  --data "$CACHE" \
  --device MAX78000 \
  --use-bias \
  --gpus 0

python train.py \
  --summary model \
  --model jestermotion12 \
  --dataset jester_motion12_hard \
  --data "$CACHE" \
  --device MAX78000 \
  --use-bias \
  --gpus 0

echo "Jester optimization sanity checks passed"
