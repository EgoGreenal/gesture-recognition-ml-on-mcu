#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT="/path/to/20bn-jester"
AI8X="$PROJECT/ai8x-training"
ENV_DIR="$PROJECT/envs/ai8x-py311"
OUT_DIR="$PROJECT/runtime/ai8x_runs"
SCRATCH_TMP="$PROJECT/tmp"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
CACHE="$PROJECT/cache/jester_cache_signedmotion12_64_full"
SOURCE_RUN="$PROJECT/runs/ai8x_jestermotion12resxconv_signed_huge_teacher_distill_20260526_155122"
STUDENT_BEST="$SOURCE_RUN/distill_artifacts/student_distilled_best.pth.tar"
BACKUP_ROOT="$PROJECT/runs/ai8x_jestermotion12resxconv_signed_huge_teacher_qat_sweep_${RUN_ID}"
LOG="$OUT_DIR/jester_huge_teacher_qat_sweep_${RUN_ID}.log"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p "$OUT_DIR" "$BACKUP_ROOT" "$SCRATCH_TMP"
export TMPDIR="$SCRATCH_TMP"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

if [ ! -f "$STUDENT_BEST" ]; then
  echo "missing student checkpoint: $STUDENT_BEST" >&2
  exit 1
fi

SWEEP_SPECS=(
  "lr2em5_ep45:0.00002:45"
  "lr1em5_ep60:0.00001:60"
  "lr5em6_ep75:0.000005:75"
  "lr1em4_ep18:0.0001:18"
)

echo "[$(date)] huge-teacher QAT sweep started" | tee "$LOG"
echo "project=$PROJECT" | tee -a "$LOG"
echo "source_run=$SOURCE_RUN" | tee -a "$LOG"
echo "student_best=$STUDENT_BEST" | tee -a "$LOG"
echo "cache=$CACHE" | tee -a "$LOG"
echo "batch_size=$BATCH_SIZE" | tee -a "$LOG"
printf 'sweep_specs=%s\n' "${SWEEP_SPECS[*]}" | tee -a "$LOG"

cd "$AI8X"

for spec in "${SWEEP_SPECS[@]}"; do
  IFS=: read -r suffix lr epochs <<< "$spec"
  name="ai8x_jestermotion12resxconv_signed_huge_teacher_qat_sweep_${RUN_ID}_${suffix}"
  echo "[$(date)] QAT sweep start suffix=$suffix lr=$lr epochs=$epochs name=$name" | tee -a "$LOG"

  python train.py \
    --lr "$lr" \
    --optimizer adam \
    --epochs "$epochs" \
    --batch-size "$BATCH_SIZE" \
    --use-bias \
    --deterministic \
    --compress policies/schedule.yaml \
    --qat-policy policies/qat_policy_jestermotion12resxconv.yaml \
    --model jestermotion12resxconv \
    --dataset jester_signedmotion12 \
    --data "$CACHE" \
    --validation-split 0 \
    --device MAX78000 \
    --gpus 0 \
    --workers 0 \
    --print-freq 200 \
    --name "$name" \
    --out-dir "$OUT_DIR" \
    --exp-load-weights-from "$STUDENT_BEST" \
    2>&1 | tee -a "$LOG"

  qat_dir="$(find "$OUT_DIR" -maxdepth 1 -type d -name "${name}*" | sort | tail -1)"
  if [ -z "$qat_dir" ]; then
    echo "[$(date)] ERROR: missing QAT output dir for $name" | tee -a "$LOG"
    exit 1
  fi
  mkdir -p "$BACKUP_ROOT/$suffix"
  cp -a "$qat_dir" "$BACKUP_ROOT/$suffix/qat_artifacts"
  echo "[$(date)] QAT sweep finished suffix=$suffix dir=$qat_dir" | tee -a "$LOG"
done

cp "$LOG" "$BACKUP_ROOT/train.log"
find "$BACKUP_ROOT" -maxdepth 4 -type f \( -name '*best*.pth.tar' -o -name '*checkpoint*.pth.tar' -o -name '*.log' \) -ls | tee -a "$LOG"
echo "[$(date)] huge-teacher QAT sweep finished backup_dir=$BACKUP_ROOT" | tee -a "$LOG"
