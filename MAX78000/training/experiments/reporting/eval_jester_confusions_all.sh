#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT="/path/to/20bn-jester"
AI8X="$PROJECT/ai8x-training"
PY="$PROJECT/envs/ai8x-py311/bin/python"
OUT="$PROJECT/runs/confusion_all_20260530"

mkdir -p "$OUT/logs"
cd "$AI8X"

run_eval() {
  local name="$1"
  local model="$2"
  local dataset="$3"
  local cache="$4"
  local policy="$5"
  local ckpt="$6"
  local log="$OUT/${name}.log"

  if grep -q "==> Confusion:" "$log" 2>/dev/null; then
    echo "[$(date)] skip existing $name"
    return
  fi

  echo "[$(date)] evaluating $name" | tee "$log"
  "$PY" train.py \
    --evaluate \
    --confusion \
    --batch-size 64 \
    --use-bias \
    --deterministic \
    --qat-policy "$policy" \
    --model "$model" \
    --dataset "$dataset" \
    --data "$cache" \
    --validation-split 0 \
    --device MAX78000 \
    --gpus 0 \
    --workers 0 \
    --print-freq 200 \
    --name "confusion_${name}" \
    --out-dir "$OUT/logs" \
    --resume-from "$ckpt" \
    2>&1 | tee -a "$log"
}

run_eval "ai8x_jesternet_qat_20260520_005821" \
  "jesternet" "jester" "$PROJECT/cache/jester_cache_raw8_64_full" \
  "policies/qat_policy_jesternet.yaml" \
  "$PROJECT/runs/ai8x_jesternet_qat_20260520_005821/artifacts/ai8x_jesternet_qat_20260520_005821_qat_best.pth.tar"

run_eval "ai8x_jester12wide_20260520_161242" \
  "jester12wide" "jester12" "$PROJECT/cache/jester_cache_raw12_64_full" \
  "policies/qat_policy_jester12wide.yaml" \
  "$PROJECT/runs/ai8x_jester12wide_20260520_161242/qat_artifacts/ai8x_jester12wide_qat_20260520_161242_qat_best.pth.tar"

run_eval "ai8x_jestermotion8_20260520_211203" \
  "jestermotion8" "jester_motion8" "/tmp/jester_cache_motion8_64_full" \
  "policies/qat_policy_jestermotion8.yaml" \
  "$PROJECT/runs/ai8x_jestermotion8_20260520_211203/qat_artifacts/ai8x_jestermotion8_qat_20260520_211203_qat_best.pth.tar"

run_eval "ai8x_jestermotion12_20260520_184836" \
  "jestermotion12" "jester_motion12" "$PROJECT/cache/jester_cache_motion12_64_full" \
  "policies/qat_policy_jestermotion12.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12_20260520_184836/qat_artifacts/ai8x_jestermotion12_qat_20260520_184836_qat_best.pth.tar"

run_eval "ai8x_jestermotion12_qat_lowlr_20260521_141542" \
  "jestermotion12" "jester_motion12" "$PROJECT/cache/jester_cache_motion12_64_full" \
  "policies/qat_policy_jestermotion12.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12_qat_lowlr_20260521_141542/qat_artifacts/ai8x_jestermotion12_qat_lowlr_20260521_141542_qat_best.pth.tar"

run_eval "ai8x_jestermixed12_20260521_141543" \
  "jestermixed12" "jester_mixed12" "$PROJECT/cache/jester_cache_mixed12_64_full" \
  "policies/qat_policy_jestermixed12.yaml" \
  "$PROJECT/runs/ai8x_jestermixed12_20260521_141543/qat_artifacts/ai8x_jestermixed12_qat_20260521_141543_qat_best.pth.tar"

run_eval "ai8x_jestersignedmotion12_20260521_174618" \
  "jestersignedmotion12" "jester_signedmotion12" "$PROJECT/cache/jester_cache_signedmotion12_64_full" \
  "policies/qat_policy_jestersignedmotion12.yaml" \
  "$PROJECT/runs/ai8x_jestersignedmotion12_20260521_174618/qat_artifacts/ai8x_jestersignedmotion12_qat_20260521_174618_qat_best.pth.tar"

run_eval "ai8x_jestermotion12_hard_20260521_192437" \
  "jestermotion12" "jester_motion12_hard" "$PROJECT/cache/jester_cache_motion12_64_full" \
  "policies/qat_policy_jestermotion12_hard.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12_hard_20260521_192437/qat_artifacts/ai8x_jestermotion12_hard_qat_20260521_192437_qat_best.pth.tar"

run_eval "ai8x_jestermotion12res_20260521_163044" \
  "jestermotion12res" "jester_motion12" "$PROJECT/cache/jester_cache_motion12_64_full" \
  "policies/qat_policy_jestermotion12res.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12res_20260521_163044/qat_artifacts/ai8x_jestermotion12res_qat_20260521_163044_qat_best.pth.tar"

run_eval "ai8x_jestermotion12res_signed_20260521_225051" \
  "jestermotion12res" "jester_signedmotion12" "$PROJECT/cache/jester_cache_signedmotion12_64_full" \
  "policies/qat_policy_jestermotion12res.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12res_signed_20260521_225051/qat_artifacts/ai8x_jestermotion12res_signed_qat_20260521_225051_qat_best.pth.tar"

run_eval "ai8x_jestermotion12resfc192_20260522_000610" \
  "jestermotion12resfc192" "jester_motion12" "$PROJECT/cache/jester_cache_motion12_64_full" \
  "policies/qat_policy_jestermotion12resfc192.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12resfc192_20260522_000610/qat_artifacts/ai8x_jestermotion12resfc192_qat_20260522_000610_qat_best.pth.tar"

run_eval "ai8x_jestermotion12resfc192_signed_20260522_101448" \
  "jestermotion12resfc192" "jester_signedmotion12" "$PROJECT/cache/jester_cache_signedmotion12_64_full" \
  "policies/qat_policy_jestermotion12resfc192.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12resfc192_signed_20260522_101448/qat_artifacts/ai8x_jestermotion12resfc192_signed_qat_20260522_101448_qat_best.pth.tar"

run_eval "ai8x_jestermotion12resxconv_20260522_012825" \
  "jestermotion12resxconv" "jester_motion12" "$PROJECT/cache/jester_cache_motion12_64_full" \
  "policies/qat_policy_jestermotion12resxconv.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12resxconv_20260522_012825/qat_artifacts/ai8x_jestermotion12resxconv_qat_20260522_012825_qat_best.pth.tar"

run_eval "ai8x_jestermotion12resxconv_signed_20260522_113043" \
  "jestermotion12resxconv" "jester_signedmotion12" "$PROJECT/cache/jester_cache_signedmotion12_64_full" \
  "policies/qat_policy_jestermotion12resxconv.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12resxconv_signed_20260522_113043/qat_artifacts/ai8x_jestermotion12resxconv_signed_qat_20260522_113043_qat_best.pth.tar"

run_eval "ai8x_jestermotion12resxconv_signed_distill_20260522_133850" \
  "jestermotion12resxconv" "jester_signedmotion12" "$PROJECT/cache/jester_cache_signedmotion12_64_full" \
  "policies/qat_policy_jestermotion12resxconv.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12resxconv_signed_distill_20260522_133850/qat_artifacts/ai8x_jestermotion12resxconv_signed_distill_qat_20260522_133850_qat_best.pth.tar"

run_eval "ai8x_jestermotion12resxconv_signed_huge_teacher_distill_20260526_155122" \
  "jestermotion12resxconv" "jester_signedmotion12" "$PROJECT/cache/jester_cache_signedmotion12_64_full" \
  "policies/qat_policy_jestermotion12resxconv.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12resxconv_signed_huge_teacher_distill_20260526_155122/qat_artifacts/ai8x_jestermotion12resxconv_signed_huge_teacher_distill_qat_20260526_155122_qat_best.pth.tar"

run_eval "ai8x_jestermotion12resxconv_signed_huge_teacher_qat_sweep_20260528_102149" \
  "jestermotion12resxconv" "jester_signedmotion12" "$PROJECT/cache/jester_cache_signedmotion12_64_full" \
  "policies/qat_policy_jestermotion12resxconv.yaml" \
  "$PROJECT/runs/ai8x_jestermotion12resxconv_signed_huge_teacher_qat_sweep_20260528_102149/lr5em6_ep75/qat_artifacts/ai8x_jestermotion12resxconv_signed_huge_teacher_qat_sweep_20260528_102149_lr5em6_ep75_qat_best.pth.tar"

"$PY" "$PROJECT/parse_jester_confusions.py" \
  --out "$OUT" \
  --labels "$PROJECT/cache/jester_cache_raw8_64_full/labels.json" \
  --log ai8x_jesternet_qat_20260520_005821="$OUT/ai8x_jesternet_qat_20260520_005821.log" \
  --log ai8x_jester12wide_20260520_161242="$OUT/ai8x_jester12wide_20260520_161242.log" \
  --log ai8x_jestermotion8_20260520_211203="$OUT/ai8x_jestermotion8_20260520_211203.log" \
  --log ai8x_jestermotion12_20260520_184836="$OUT/ai8x_jestermotion12_20260520_184836.log" \
  --log ai8x_jestermotion12_qat_lowlr_20260521_141542="$OUT/ai8x_jestermotion12_qat_lowlr_20260521_141542.log" \
  --log ai8x_jestermixed12_20260521_141543="$OUT/ai8x_jestermixed12_20260521_141543.log" \
  --log ai8x_jestersignedmotion12_20260521_174618="$OUT/ai8x_jestersignedmotion12_20260521_174618.log" \
  --log ai8x_jestermotion12_hard_20260521_192437="$OUT/ai8x_jestermotion12_hard_20260521_192437.log" \
  --log ai8x_jestermotion12res_20260521_163044="$OUT/ai8x_jestermotion12res_20260521_163044.log" \
  --log ai8x_jestermotion12res_signed_20260521_225051="$OUT/ai8x_jestermotion12res_signed_20260521_225051.log" \
  --log ai8x_jestermotion12resfc192_20260522_000610="$OUT/ai8x_jestermotion12resfc192_20260522_000610.log" \
  --log ai8x_jestermotion12resfc192_signed_20260522_101448="$OUT/ai8x_jestermotion12resfc192_signed_20260522_101448.log" \
  --log ai8x_jestermotion12resxconv_20260522_012825="$OUT/ai8x_jestermotion12resxconv_20260522_012825.log" \
  --log ai8x_jestermotion12resxconv_signed_20260522_113043="$OUT/ai8x_jestermotion12resxconv_signed_20260522_113043.log" \
  --log ai8x_jestermotion12resxconv_signed_distill_20260522_133850="$OUT/ai8x_jestermotion12resxconv_signed_distill_20260522_133850.log" \
  --log ai8x_jestermotion12resxconv_signed_huge_teacher_distill_20260526_155122="$OUT/ai8x_jestermotion12resxconv_signed_huge_teacher_distill_20260526_155122.log" \
  --log ai8x_jestermotion12resxconv_signed_huge_teacher_qat_sweep_20260528_102149="$OUT/ai8x_jestermotion12resxconv_signed_huge_teacher_qat_sweep_20260528_102149.log"

echo "[$(date)] all deployable-model confusion matrices complete: $OUT"
