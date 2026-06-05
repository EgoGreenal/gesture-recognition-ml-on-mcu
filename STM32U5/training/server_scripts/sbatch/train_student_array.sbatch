#!/bin/bash
#SBATCH --job-name=train_student_arr
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
# 1-GPU partial-node per task (Plan Day 9-10 strategy: student is small enough
# that DDP overhead > compute gain; 7 variants run in parallel as separate
# array tasks, each on its own GPU. Mixed-state partial-node queue typically
# ahead ~30min — full array completes wall ≈ max(per-variant) ≈ 60-90 min.)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-6%7
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/student_arr_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/student_arr_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1   # flush stdout/err to SLURM log immediately

VARIANTS=(V0 V1 V2 V3 V4 V5 V6)
V="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"

echo "== node info =="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}  variant=${V}"
nvidia-smi
echo

# job_id distinct per array task so per-variant out dir doesn't collide
JOB_ID="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"

python "${PROJECT_ROOT}/scripts/train_student.py" \
    --variant "${V}" \
    --soft_labels_ta "${PROJECT_ROOT}/soft_labels/soft_labels_ta_train.npy" \
    --epochs 8 \
    --warmup_epochs 1 \
    --batch_size 64 \
    --lr 1e-3 \
    --wd 1e-4 \
    --alpha 0.3 \
    --temperature 4.0 \
    --frame_weights 0.25 0.25 0.25 0.25 \
    --num_workers 8 \
    --max_steps 700 \
    --fast_val_batches 10 \
    --val_every 0 \
    --seed 42 \
    --log_every 100 \
    --job_id "${JOB_ID}"
# Day 9 throughput calibration (see daily log):
#   * Pipeline-L generator is GIL-bound (~9 samples/sec aggregate even at
#     num_shards=8). At bs=64 that's ~7s/batch.
#   * Plan budgeted 20-30 epochs × full-set passes; realistic budget on this
#     pipeline is ~5h wall per variant. 8 epochs × max_steps=1000 caps work
#     at ~8000 batches × 7s = ~15h wall — STILL TOO LONG, see note below.
#
# Fallback strategy in case of SLURM timeout:
#   * train script saves best.weights.h5 + last.weights.h5 every epoch and
#     metrics.json incrementally. If the 4h wall hits mid-training, the
#     latest best checkpoint is recoverable; analyze_students.py just picks
#     the most-recent run dir.
#
# Each epoch supervises 1000 × 64 = 64000 samples ≈ 54% of train set per
# epoch (with shuffle, accumulates across epochs).
