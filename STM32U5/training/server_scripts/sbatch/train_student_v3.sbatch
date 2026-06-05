#!/bin/bash
#SBATCH --job-name=train_student_v3
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 10 v3: depth scan + KD ablation
#   * Day 10 v2 array (41422954) found V4 (5-block) ≫ V5 (3-block, 2.3× width)
#     → TSM needs temporal depth. Sweep depth at 6/7/8 blocks (V7/V8/V9).
#   * Sanity: V4 with alpha=0.0 (pure KD) and alpha=1.0 (pure CE / hard-label
#     only) to verify the KD signal is actually contributing the +X% over
#     hard-label baseline.
#
# Array spec (5 tasks):
#   0 → V7   alpha=0.3   depth=6  (84K params)
#   1 → V8   alpha=0.3   depth=7  (167K params)
#   2 → V9   alpha=0.3   depth=8  (250K params)
#   3 → V4   alpha=0.0   pure KD baseline
#   4 → V4   alpha=1.0   pure CE / hard-label only baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%5
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/student_v3_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/student_v3_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

# task_id -> (variant, alpha)
TASKS=(
    "V7 0.3"
    "V8 0.3"
    "V9 0.3"
    "V4 0.0"
    "V4 1.0"
)
SPEC="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
V=$(echo "${SPEC}" | awk '{print $1}')
ALPHA=$(echo "${SPEC}" | awk '{print $2}')

echo "== node info =="
hostname
echo "ARRAY_TASK=${SLURM_ARRAY_TASK_ID}  variant=${V}  alpha=${ALPHA}"
nvidia-smi
echo

# Suffix job_id with alpha so V4 alpha=0.0/1.0 don't clobber each other or
# the existing alpha=0.3 ckpt from array v2.
JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_a${ALPHA}"

python "${PROJECT_ROOT}/scripts/train_student.py" \
    --variant "${V}" \
    --soft_labels_ta "${PROJECT_ROOT}/soft_labels/soft_labels_ta_train.npy" \
    --epochs 30 \
    --warmup_epochs 2 \
    --batch_size 64 \
    --lr 1e-3 \
    --wd 1e-4 \
    --alpha "${ALPHA}" \
    --temperature 4.0 \
    --frame_weights 0.10 0.20 0.30 0.40 \
    --num_workers 8 \
    --max_steps 0 \
    --fast_val_batches 30 \
    --val_every 3 \
    --seed 42 \
    --log_every 200 \
    --job_id "${JOB_ID}"
