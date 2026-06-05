#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
UPLOAD_DIR="$PROJECT/codex_uploads/jestermotion8"
AI8X="$PROJECT/ai8x-training"

mkdir -p "$AI8X/models" "$AI8X/datasets" "$AI8X/policies"

cp "$UPLOAD_DIR/jestermotion8.py" "$AI8X/models/jestermotion8.py"
cp "$UPLOAD_DIR/jester_motion8.py" "$AI8X/datasets/jester_motion8.py"
cp "$UPLOAD_DIR/qat_policy_jestermotion8.yaml" "$AI8X/policies/qat_policy_jestermotion8.yaml"
cp "$UPLOAD_DIR/remote_run_ai8x_jestermotion8_qat.sh" "$PROJECT/remote_run_ai8x_jestermotion8_qat.sh"
cp "$UPLOAD_DIR/remote_start_ai8x_jestermotion8_tmux.sh" "$PROJECT/remote_start_ai8x_jestermotion8_tmux.sh"
cp "$UPLOAD_DIR/remote_ai8x_jestermotion8_sanity.sh" "$PROJECT/remote_ai8x_jestermotion8_sanity.sh"

sed -i 's/\r$//' \
  "$PROJECT/remote_run_ai8x_jestermotion8_qat.sh" \
  "$PROJECT/remote_start_ai8x_jestermotion8_tmux.sh" \
  "$PROJECT/remote_ai8x_jestermotion8_sanity.sh"

chmod +x \
  "$PROJECT/remote_run_ai8x_jestermotion8_qat.sh" \
  "$PROJECT/remote_start_ai8x_jestermotion8_tmux.sh" \
  "$PROJECT/remote_ai8x_jestermotion8_sanity.sh"

echo "installed jestermotion8 files"
find "$AI8X/models" "$AI8X/datasets" "$AI8X/policies" -maxdepth 1 \
  \( -name 'jestermotion8.py' -o -name 'jester_motion8.py' -o -name 'qat_policy_jestermotion8.yaml' \) \
  -type f -ls
