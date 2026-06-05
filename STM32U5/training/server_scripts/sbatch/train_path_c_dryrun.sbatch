#!/bin/bash
#SBATCH --job-name=path_c_dry
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 10 Path C smoke test: pick one variant from each family, 1 epoch,
# max_steps=30 -- verifies the end-to-end train_step works for sequence
# (C1/C3) + stacked (C2) modes before launching the 9-variant array.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --array=0-2
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/path_c_dry_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/path_c_dry_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

VARIANTS=("C1a" "C2a" "C3a")
V="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
echo "== dry-run task ${SLURM_ARRAY_TASK_ID}: variant ${V} =="

JOB_ID="dry_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

python "${PROJECT_ROOT}/scripts/train_student.py" \
    --variant "${V}" \
    --soft_labels_ta "${PROJECT_ROOT}/soft_labels/soft_labels_ta_train.npy" \
    --epochs 1 \
    --warmup_epochs 0 \
    --batch_size 64 \
    --lr 1e-3 \
    --wd 1e-4 \
    --alpha 0.3 \
    --temperature 4.0 \
    --frame_weights 0.10 0.20 0.30 0.40 \
    --num_workers 8 \
    --max_steps 30 \
    --fast_val_batches 5 \
    --val_every 1 \
    --seed 42 \
    --log_every 5 \
    --job_id "${JOB_ID}"
