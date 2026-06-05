#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
UPLOAD_DIR="$PROJECT/codex_uploads/jestermotion12"
AI8X="$PROJECT/ai8x-training"

mkdir -p "$AI8X/models" "$AI8X/datasets" "$AI8X/policies"

cp "$UPLOAD_DIR/jestermotion12.py" "$AI8X/models/jestermotion12.py"
cp "$UPLOAD_DIR/jester_motion12.py" "$AI8X/datasets/jester_motion12.py"
cp "$UPLOAD_DIR/qat_policy_jestermotion12.yaml" "$AI8X/policies/qat_policy_jestermotion12.yaml"
cp "$UPLOAD_DIR/train_jester_cached.py" "$PROJECT/train_jester_cached.py"
cp "$UPLOAD_DIR/remote_run_ai8x_jestermotion12_qat.sh" "$PROJECT/remote_run_ai8x_jestermotion12_qat.sh"
cp "$UPLOAD_DIR/remote_start_ai8x_jestermotion12_tmux.sh" "$PROJECT/remote_start_ai8x_jestermotion12_tmux.sh"
cp "$UPLOAD_DIR/remote_ai8x_jestermotion12_sanity.sh" "$PROJECT/remote_ai8x_jestermotion12_sanity.sh"

sed -i 's/\r$//' \
  "$PROJECT/remote_run_ai8x_jestermotion12_qat.sh" \
  "$PROJECT/remote_start_ai8x_jestermotion12_tmux.sh" \
  "$PROJECT/remote_ai8x_jestermotion12_sanity.sh" \
  "$PROJECT/train_jester_cached.py"

chmod +x \
  "$PROJECT/remote_run_ai8x_jestermotion12_qat.sh" \
  "$PROJECT/remote_start_ai8x_jestermotion12_tmux.sh" \
  "$PROJECT/remote_ai8x_jestermotion12_sanity.sh" \
  "$PROJECT/train_jester_cached.py"

echo "installed jestermotion12 files"
find "$AI8X/models" "$AI8X/datasets" "$AI8X/policies" -maxdepth 1 \
  \( -name 'jestermotion12.py' -o -name 'jester_motion12.py' -o -name 'qat_policy_jestermotion12.yaml' \) \
  -type f -ls
