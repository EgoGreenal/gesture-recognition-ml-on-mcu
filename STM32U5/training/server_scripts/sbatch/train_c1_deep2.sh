#!/bin/bash
#SBATCH --job-name=c1_deep2
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 10 v3 deep dive round 2: fill the "deployment-friendly + 85%+ accuracy"
# sweet spot.
#   0 -> C1k  w=0.60, 17blk, 64x64, 30ep   (970K params, ~947 KB INT8 Flash)
#   1 -> C1m  w=0.50, 17blk, 80x80, 30ep   (722K params, 705 KB INT8)
#   2 -> C1n  w=1.00, 13blk, 64x64, 30ep   (702K params, 685 KB INT8)
#   3 -> C1o  w=0.80, 17blk, 64x64, 30ep   (1.55M params, 1509 KB INT8)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/c1deep2_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/c1deep2_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

VARIANTS=("C1k" "C1m" "C1n" "C1o")
V="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"

echo "== node info =="; hostname; echo "task=${SLURM_ARRAY_TASK_ID} variant=${V}"
nvidia-smi; echo

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
