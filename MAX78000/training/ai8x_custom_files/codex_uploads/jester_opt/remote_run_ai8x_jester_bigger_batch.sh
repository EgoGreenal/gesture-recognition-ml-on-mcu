#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
OUT_DIR="$PROJECT/runtime/ai8x_runs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_DIR/jester_bigger_batch_${RUN_ID}.log"

mkdir -p "$OUT_DIR"

run_variant() {
  local label="$1"
  shift
  echo "[$(date)] START $label" | tee -a "$LOG"
  (
    cd "$PROJECT"
    "$@"
  ) 2>&1 | tee -a "$LOG"
  local status="${PIPESTATUS[0]}"
  echo "[$(date)] END $label status=$status" | tee -a "$LOG"
  return "$status"
}

echo "[$(date)] Jester bigger-model batch started" | tee "$LOG"

run_variant "residual + signed-motion" env \
  MODEL=jestermotion12res \
  DATASET=jester_signedmotion12 \
  CACHE="$PROJECT/cache/jester_cache_signedmotion12_64_full" \
  CACHE_ENCODING=signed_motion \
  RUN_SLUG=ai8x_jestermotion12res_signed \
  QAT_POLICY=qat_policy_jestermotion12res.yaml \
  "$PROJECT/remote_run_ai8x_jester_variant_qat.sh"

run_variant "residual fc192" env \
  MODEL=jestermotion12resfc192 \
  DATASET=jester_motion12 \
  CACHE="$PROJECT/cache/jester_cache_motion12_64_full" \
  CACHE_ENCODING=motion \
  RUN_SLUG=ai8x_jestermotion12resfc192 \
  QAT_POLICY=qat_policy_jestermotion12resfc192.yaml \
  "$PROJECT/remote_run_ai8x_jester_variant_qat.sh"

run_variant "residual extra low-res conv" env \
  MODEL=jestermotion12resxconv \
  DATASET=jester_motion12 \
  CACHE="$PROJECT/cache/jester_cache_motion12_64_full" \
  CACHE_ENCODING=motion \
  RUN_SLUG=ai8x_jestermotion12resxconv \
  QAT_POLICY=qat_policy_jestermotion12resxconv.yaml \
  "$PROJECT/remote_run_ai8x_jester_variant_qat.sh"

echo "[$(date)] Jester bigger-model batch finished" | tee -a "$LOG"
