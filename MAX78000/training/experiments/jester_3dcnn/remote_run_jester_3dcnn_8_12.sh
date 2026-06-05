#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/20bn-jester"
PY="$ROOT/envs/ai8x-py311/bin/python"
DATASET="/path/to/20bn-jester-v1-complete"
ANNOTATIONS="$ROOT/annotations"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/runs/jester_3dcnn_raw8_raw12_${STAMP}"
LOG_DIR="$ROOT/logs"
EPOCHS="${JESTER_3DCNN_EPOCHS:-60}"
BATCH_SIZE="${JESTER_3DCNN_BATCH_SIZE:-64}"
WORKERS="${JESTER_3DCNN_WORKERS:-4}"

mkdir -p "$RUN_ROOT" "$LOG_DIR"
echo "$RUN_ROOT" > "$LOG_DIR/jester_3dcnn_latest_run_dir.txt"

echo "run_root=$RUN_ROOT"
echo "epochs=$EPOCHS batch_size=$BATCH_SIZE workers=$WORKERS"
echo "note=true Conv3D is not an ai8x/MAX78000 synthesizable operator; this run is a software accuracy/PTQ probe."

for FRAMES in 8 12; do
  CACHE="$ROOT/cache/jester_cache_raw${FRAMES}_64_full"
  OUT="$RUN_ROOT/raw${FRAMES}"
  if [[ ! -f "$CACHE/train.pt" || ! -f "$CACHE/val.pt" ]]; then
    echo "building_cache frames=$FRAMES cache=$CACHE"
    "$PY" "$ROOT/train_jester_cached.py" cache \
      --dataset-root "$DATASET" \
      --annotations-dir "$ANNOTATIONS" \
      --cache-dir "$CACHE" \
      --frames "$FRAMES" \
      --image-size 64 \
      --encoding raw \
      --train-samples 0 \
      --val-samples 0 \
      --seed 123
  else
    echo "cache_exists frames=$FRAMES cache=$CACHE"
  fi

  echo "training_3dcnn frames=$FRAMES out=$OUT"
  "$PY" "$ROOT/train_jester_3dcnn.py" \
    --cache-dir "$CACHE" \
    --out-dir "$OUT" \
    --frames "$FRAMES" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --label-smoothing 0.05 \
    --width-mult 1.0 \
    --dropout 0.25 \
    --print-freq 200 \
    --seed 123
done

echo "DONE run_root=$RUN_ROOT"
