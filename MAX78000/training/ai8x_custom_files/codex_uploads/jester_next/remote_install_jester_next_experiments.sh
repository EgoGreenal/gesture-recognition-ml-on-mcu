#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
UPLOAD_DIR="$PROJECT/codex_uploads/jester_next"
AI8X="$PROJECT/ai8x-training"

mkdir -p "$AI8X/models" "$AI8X/datasets" "$AI8X/policies"

cp "$UPLOAD_DIR/jestermixed12.py" "$AI8X/models/jestermixed12.py"
cp "$UPLOAD_DIR/jester_mixed12.py" "$AI8X/datasets/jester_mixed12.py"
cp "$UPLOAD_DIR/qat_policy_jestermixed12.yaml" "$AI8X/policies/qat_policy_jestermixed12.yaml"
cp "$UPLOAD_DIR/train_jester_cached.py" "$PROJECT/train_jester_cached.py"
cp "$UPLOAD_DIR/remote_run_ai8x_jestermotion12_qat_lowlr.sh" "$PROJECT/remote_run_ai8x_jestermotion12_qat_lowlr.sh"
cp "$UPLOAD_DIR/remote_start_ai8x_jestermotion12_qat_lowlr_tmux.sh" "$PROJECT/remote_start_ai8x_jestermotion12_qat_lowlr_tmux.sh"
cp "$UPLOAD_DIR/remote_run_ai8x_jestermixed12_qat.sh" "$PROJECT/remote_run_ai8x_jestermixed12_qat.sh"
cp "$UPLOAD_DIR/remote_start_ai8x_jestermixed12_tmux.sh" "$PROJECT/remote_start_ai8x_jestermixed12_tmux.sh"

sed -i 's/\r$//' \
  "$PROJECT/train_jester_cached.py" \
  "$PROJECT/remote_run_ai8x_jestermotion12_qat_lowlr.sh" \
  "$PROJECT/remote_start_ai8x_jestermotion12_qat_lowlr_tmux.sh" \
  "$PROJECT/remote_run_ai8x_jestermixed12_qat.sh" \
  "$PROJECT/remote_start_ai8x_jestermixed12_tmux.sh"

chmod +x \
  "$PROJECT/train_jester_cached.py" \
  "$PROJECT/remote_run_ai8x_jestermotion12_qat_lowlr.sh" \
  "$PROJECT/remote_start_ai8x_jestermotion12_qat_lowlr_tmux.sh" \
  "$PROJECT/remote_run_ai8x_jestermixed12_qat.sh" \
  "$PROJECT/remote_start_ai8x_jestermixed12_tmux.sh"

echo "installed next Jester experiment files"
find "$AI8X/models" "$AI8X/datasets" "$AI8X/policies" -maxdepth 1 \
  \( -name 'jestermixed12.py' -o -name 'jester_mixed12.py' -o -name 'qat_policy_jestermixed12.yaml' \) \
  -type f -ls
