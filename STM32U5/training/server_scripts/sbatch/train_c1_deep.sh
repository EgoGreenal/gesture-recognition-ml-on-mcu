#!/bin/bash
#SBATCH --job-name=c1_deep
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 10 v2 deep dive: push C1 family from 84% (C1b) to 85-90%.
# Sweep width × input resolution × epoch budget.
#
#   0 -> C1d  w=0.35, 17blk, 64x64, 30ep  (430K)
#   1 -> C1e  w=0.40, 17blk, 64x64, 30ep  (524K)
#   2 -> C1f  w=0.75, 17blk, 64x64, 30ep  (1.39M)
#   3 -> C1g  w=1.00, 17blk, 64x64, 30ep  (2.26M)  ← biggest, may exceed Flash
#   4 -> C1h  w=0.50, 17blk, 96x96, 30ep  (722K)
#   5 -> C1i  w=0.50, 17blk, 112x112, 30ep (722K) ← TA-resolution input
#   6 -> C1j  w=0.50, 17blk, 64x64, 60ep  (722K) ← C1b retrain longer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --array=0-6%7
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/c1deep_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/c1deep_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

# (variant, epochs)
TASKS=(
    "C1d 30"
    "C1e 30"
    "C1f 30"
    "C1g 30"
    "C1h 30"
    "C1i 30"
    "C1j 60"
)
SPEC="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
V=$(echo "${SPEC}" | awk '{print $1}')
EPOCHS=$(echo "${SPEC}" | awk '{print $2}')

echo "== node info =="; hostname; echo "task=${SLURM_ARRAY_TASK_ID} variant=${V} epochs=${EPOCHS}"
nvidia-smi; echo

JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

python "${PROJECT_ROOT}/scripts/train_student.py" \
    --variant "${V}" \
    --soft_labels_ta "${PROJECT_ROOT}/soft_labels/soft_labels_ta_train.npy" \
    --epochs "${EPOCHS}" \
    --warmup_epochs 2 \
    --batch_size 64 \
    --lr 1e-3 \
    --wd 1e-4 \
    --alpha 0.3 \
    --temperature 4.0 \
    --frame_weights 0.10 0.20 0.30 0.40 \
    --num_workers 8 \
    --max_steps 0 \
    --fast_val_batches 30 \
    --val_every 5 \
    --seed 42 \
    --log_every 200 \
    --job_id "${JOB_ID}"
