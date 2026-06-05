#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
UPLOAD_DIR="$PROJECT/codex_uploads/jester_opt"
AI8X="$PROJECT/ai8x-training"

mkdir -p "$AI8X/models" "$AI8X/datasets" "$AI8X/policies"

cp "$UPLOAD_DIR/jestermotion12res.py" "$AI8X/models/jestermotion12res.py"
cp "$UPLOAD_DIR/jestermotion12resfc192.py" "$AI8X/models/jestermotion12resfc192.py"
cp "$UPLOAD_DIR/jestermotion12resxconv.py" "$AI8X/models/jestermotion12resxconv.py"
cp "$UPLOAD_DIR/jestersignedmotion12.py" "$AI8X/models/jestersignedmotion12.py"
cp "$UPLOAD_DIR/jester_signedmotion12.py" "$AI8X/datasets/jester_signedmotion12.py"
cp "$UPLOAD_DIR/jester_motion12_hard.py" "$AI8X/datasets/jester_motion12_hard.py"
cp "$UPLOAD_DIR/qat_policy_jestermotion12res.yaml" "$AI8X/policies/qat_policy_jestermotion12res.yaml"
cp "$UPLOAD_DIR/qat_policy_jestermotion12resfc192.yaml" "$AI8X/policies/qat_policy_jestermotion12resfc192.yaml"
cp "$UPLOAD_DIR/qat_policy_jestermotion12resxconv.yaml" "$AI8X/policies/qat_policy_jestermotion12resxconv.yaml"
cp "$UPLOAD_DIR/qat_policy_jestersignedmotion12.yaml" "$AI8X/policies/qat_policy_jestersignedmotion12.yaml"
cp "$UPLOAD_DIR/qat_policy_jestermotion12_hard.yaml" "$AI8X/policies/qat_policy_jestermotion12_hard.yaml"
cp "$UPLOAD_DIR/train_jester_cached.py" "$PROJECT/train_jester_cached.py"
cp "$UPLOAD_DIR/remote_run_ai8x_jester_variant_qat.sh" "$PROJECT/remote_run_ai8x_jester_variant_qat.sh"
cp "$UPLOAD_DIR/remote_run_ai8x_jester_bigger_batch.sh" "$PROJECT/remote_run_ai8x_jester_bigger_batch.sh"
cp "$UPLOAD_DIR/remote_start_ai8x_jester_bigger_batch_tmux.sh" "$PROJECT/remote_start_ai8x_jester_bigger_batch_tmux.sh"
cp "$UPLOAD_DIR/remote_run_ai8x_jestermotion12res_qat.sh" "$PROJECT/remote_run_ai8x_jestermotion12res_qat.sh"
cp "$UPLOAD_DIR/remote_run_ai8x_jestersignedmotion12_qat.sh" "$PROJECT/remote_run_ai8x_jestersignedmotion12_qat.sh"
cp "$UPLOAD_DIR/remote_run_ai8x_jestermotion12_hard_qat.sh" "$PROJECT/remote_run_ai8x_jestermotion12_hard_qat.sh"
cp "$UPLOAD_DIR/remote_run_ai8x_jester_opt_batch.sh" "$PROJECT/remote_run_ai8x_jester_opt_batch.sh"
cp "$UPLOAD_DIR/remote_start_ai8x_jester_opt_batch_tmux.sh" "$PROJECT/remote_start_ai8x_jester_opt_batch_tmux.sh"
cp "$UPLOAD_DIR/remote_ai8x_jester_opt_sanity.sh" "$PROJECT/remote_ai8x_jester_opt_sanity.sh"

sed -i 's/\r$//' \
  "$PROJECT/train_jester_cached.py" \
  "$PROJECT/remote_run_ai8x_jester_variant_qat.sh" \
  "$PROJECT/remote_run_ai8x_jester_bigger_batch.sh" \
  "$PROJECT/remote_start_ai8x_jester_bigger_batch_tmux.sh" \
  "$PROJECT/remote_run_ai8x_jestermotion12res_qat.sh" \
  "$PROJECT/remote_run_ai8x_jestersignedmotion12_qat.sh" \
  "$PROJECT/remote_run_ai8x_jestermotion12_hard_qat.sh" \
  "$PROJECT/remote_run_ai8x_jester_opt_batch.sh" \
  "$PROJECT/remote_start_ai8x_jester_opt_batch_tmux.sh" \
  "$PROJECT/remote_ai8x_jester_opt_sanity.sh"

chmod +x \
  "$PROJECT/train_jester_cached.py" \
  "$PROJECT/remote_run_ai8x_jester_variant_qat.sh" \
  "$PROJECT/remote_run_ai8x_jester_bigger_batch.sh" \
  "$PROJECT/remote_start_ai8x_jester_bigger_batch_tmux.sh" \
  "$PROJECT/remote_run_ai8x_jestermotion12res_qat.sh" \
  "$PROJECT/remote_run_ai8x_jestersignedmotion12_qat.sh" \
  "$PROJECT/remote_run_ai8x_jestermotion12_hard_qat.sh" \
  "$PROJECT/remote_run_ai8x_jester_opt_batch.sh" \
  "$PROJECT/remote_start_ai8x_jester_opt_batch_tmux.sh" \
  "$PROJECT/remote_ai8x_jester_opt_sanity.sh"

echo "installed Jester optimization experiment files"
find "$AI8X/models" "$AI8X/datasets" "$AI8X/policies" -maxdepth 1 \
  \( -name 'jestermotion12res.py' -o -name 'jestermotion12resfc192.py' \
     -o -name 'jestermotion12resxconv.py' -o -name 'jestersignedmotion12.py' \
     -o -name 'jester_signedmotion12.py' -o -name 'jester_motion12_hard.py' \
     -o -name 'qat_policy_jestermotion12res.yaml' \
     -o -name 'qat_policy_jestermotion12resfc192.yaml' \
     -o -name 'qat_policy_jestermotion12resxconv.yaml' \
     -o -name 'qat_policy_jestersignedmotion12.yaml' \
     -o -name 'qat_policy_jestermotion12_hard.yaml' \) \
  -type f -ls
