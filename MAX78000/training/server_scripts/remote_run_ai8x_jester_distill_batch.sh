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
DISTILL_DIR="$PROJECT/runs/jester_distill_${RUN_ID}"
BACKUP_ROOT="$PROJECT/runs/ai8x_jestermotion12resxconv_signed_distill_${RUN_ID}"
LOG="$OUT_DIR/jester_distill_${RUN_ID}.log"
QAT_NAME="ai8x_jestermotion12resxconv_signed_distill_qat_${RUN_ID}"

TEACHER_EPOCHS="${TEACHER_EPOCHS:-80}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-50}"
QAT_EPOCHS="${QAT_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STUDENT_INIT="${STUDENT_INIT:-$PROJECT/runs/ai8x_jestermotion12resxconv_signed_20260522_113043/fp_artifacts/ai8x_jestermotion12resxconv_signed_fp_20260522_113043_best.pth.tar}"

mkdir -p "$OUT_DIR" "$DISTILL_DIR" "$BACKUP_ROOT" "$SCRATCH_TMP"
export TMPDIR="$SCRATCH_TMP"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

echo "[$(date)] Jester teacher distillation batch started" | tee "$LOG"
echo "project=$PROJECT" | tee -a "$LOG"
echo "cache=$CACHE" | tee -a "$LOG"
echo "teacher_epochs=$TEACHER_EPOCHS student_epochs=$STUDENT_EPOCHS qat_epochs=$QAT_EPOCHS" | tee -a "$LOG"
echo "student_init=$STUDENT_INIT" | tee -a "$LOG"

python "$PROJECT/train_jester_distill.py" \
  --cache-dir "$CACHE" \
  --ai8x-training "$AI8X" \
  --out-dir "$DISTILL_DIR" \
  --student-init "$STUDENT_INIT" \
  --teacher-epochs "$TEACHER_EPOCHS" \
  --student-epochs "$STUDENT_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --print-freq 200 \
  2>&1 | tee -a "$LOG"

STUDENT_BEST="$DISTILL_DIR/student_distilled_best.pth.tar"
if [ ! -f "$STUDENT_BEST" ]; then
  echo "[$(date)] ERROR: missing distilled student checkpoint: $STUDENT_BEST" | tee -a "$LOG"
  exit 1
fi
echo "[$(date)] student_best=$STUDENT_BEST" | tee -a "$LOG"

cd "$AI8X"

echo "[$(date)] distilled student QAT fine-tune" | tee -a "$LOG"
python train.py \
  --lr 0.00005 \
  --optimizer adam \
  --epochs "$QAT_EPOCHS" \
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
  --name "$QAT_NAME" \
  --out-dir "$OUT_DIR" \
  --exp-load-weights-from "$STUDENT_BEST" \
  2>&1 | tee -a "$LOG"

QAT_DIR="$(find "$OUT_DIR" -maxdepth 1 -type d -name "${QAT_NAME}*" | sort | tail -1)"
if [ -z "$QAT_DIR" ]; then
  echo "[$(date)] ERROR: missing QAT output dir for $QAT_NAME" | tee -a "$LOG"
  exit 1
fi

cp -a "$DISTILL_DIR" "$BACKUP_ROOT/distill_artifacts"
cp -a "$QAT_DIR" "$BACKUP_ROOT/qat_artifacts"
cp "$LOG" "$BACKUP_ROOT/train.log"
find "$BACKUP_ROOT" -maxdepth 4 -type f \( -name '*best*.pth.tar' -o -name '*checkpoint*.pth.tar' -o -name '*.log' -o -name '*.json' \) -ls | tee -a "$LOG"

echo "[$(date)] Jester teacher distillation batch finished backup_dir=$BACKUP_ROOT" | tee -a "$LOG"
