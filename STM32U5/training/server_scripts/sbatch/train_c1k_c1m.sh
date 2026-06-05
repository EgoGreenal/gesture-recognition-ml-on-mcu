#!/bin/bash
#SBATCH --job-name=c1_km
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 10 v3 round 2 retry: C1k and C1m hung on lrdn1508 at step 600 epoch 0
# (both jobs same node, same timestamp -> node-level issue, not arch bug).
# Resubmit excluding lrdn1508 to avoid same fate.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --exclude=lrdn1508
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/c1km_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/c1km_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

VARIANTS=("C1k" "C1m")
V="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
echo "task ${SLURM_ARRAY_TASK_ID} variant=${V} node=$(hostname)"
nvidia-smi

JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

python "${PROJECT_ROOT}/scripts/train_student.py" \
    --variant "${V}" \
    --soft_labels_ta "${PROJECT_ROOT}/soft_labels/soft_labels_ta_train.npy" \
    --epochs 30 \
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
