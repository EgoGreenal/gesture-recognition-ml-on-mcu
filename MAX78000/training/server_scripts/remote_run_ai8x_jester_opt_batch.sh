#!/usr/bin/env bash
set -uo pipefail

PROJECT="/path/to/20bn-jester"
OUT_DIR="$PROJECT/runtime/ai8x_runs"
BATCH_ID="$(date +%Y%m%d_%H%M%S)"
BATCH_LOG="$OUT_DIR/jester_opt_batch_${BATCH_ID}.log"

mkdir -p "$OUT_DIR"

echo "[$(date)] starting Jester optimization batch" | tee "$BATCH_LOG"
echo "batch_id=$BATCH_ID" | tee -a "$BATCH_LOG"
echo "The server has one GPU, so experiments run as a queue in this tmux session." | tee -a "$BATCH_LOG"

for script in \
  remote_run_ai8x_jestermotion12res_qat.sh \
  remote_run_ai8x_jestersignedmotion12_qat.sh \
  remote_run_ai8x_jestermotion12_hard_qat.sh
do
  echo "[$(date)] START $script" | tee -a "$BATCH_LOG"
  bash "$PROJECT/$script" 2>&1 | tee -a "$BATCH_LOG"
  status=${PIPESTATUS[0]}
  echo "[$(date)] END $script status=$status" | tee -a "$BATCH_LOG"
done

echo "[$(date)] Jester optimization batch finished" | tee -a "$BATCH_LOG"
