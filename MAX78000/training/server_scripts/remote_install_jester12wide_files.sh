#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
UPLOAD_DIR="$PROJECT/codex_uploads/jester12wide"
AI8X="$PROJECT/ai8x-training"
BASELINE_DIR="$PROJECT/baselines/jesternet8_qat_20260520"

mkdir -p "$AI8X/models" "$AI8X/datasets" "$AI8X/policies" "$BASELINE_DIR"

cp "$UPLOAD_DIR/jester12wide.py" "$AI8X/models/jester12wide.py"
cp "$UPLOAD_DIR/jester12.py" "$AI8X/datasets/jester12.py"
cp "$UPLOAD_DIR/qat_policy_jester12wide.yaml" "$AI8X/policies/qat_policy_jester12wide.yaml"
cp "$UPLOAD_DIR/train_jester_cached.py" "$PROJECT/train_jester_cached.py"
cp "$UPLOAD_DIR/remote_run_ai8x_jester12wide_qat.sh" "$PROJECT/remote_run_ai8x_jester12wide_qat.sh"
cp "$UPLOAD_DIR/remote_start_ai8x_jester12wide_tmux.sh" "$PROJECT/remote_start_ai8x_jester12wide_tmux.sh"
cp "$UPLOAD_DIR/jester_baseline_8f64_qat_20260520.md" "$BASELINE_DIR/BASELINE.md"

sed -i 's/\r$//' \
  "$PROJECT/remote_run_ai8x_jester12wide_qat.sh" \
  "$PROJECT/remote_start_ai8x_jester12wide_tmux.sh" \
  "$PROJECT/train_jester_cached.py"

chmod +x \
  "$PROJECT/remote_run_ai8x_jester12wide_qat.sh" \
  "$PROJECT/remote_start_ai8x_jester12wide_tmux.sh" \
  "$PROJECT/train_jester_cached.py"

BASELINE_CKPT="$PROJECT/runs/ai8x_jesternet_qat_20260520_005821/artifacts/ai8x_jesternet_qat_20260520_005821_qat_best.pth.tar"
if [ -f "$BASELINE_CKPT" ]; then
  cp -n "$BASELINE_CKPT" "$BASELINE_DIR/"
fi

echo "installed jester12wide files"
find "$AI8X/models" "$AI8X/datasets" "$AI8X/policies" "$BASELINE_DIR" -maxdepth 1 \
  \( -name 'jester12wide.py' -o -name 'jester12.py' -o -name 'qat_policy_jester12wide.yaml' -o -name 'BASELINE.md' -o -name '*qat_best.pth.tar' \) \
  -type f -ls
