#!/bin/bash
#SBATCH --job-name=path_c_array
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 10 Path C production training: 9 variants in parallel.
# Each variant ~1.5-2.5h wall on 1 GPU (per Plan estimate); lprod 96h cap is
# plenty of buffer. boost_qos_lprod prio 40 same as normal but the Lustre file
# locking we disabled in data_loader_tf scales well across 9 readers.
#
# Array spec (9 tasks):
#   0..2 -> C1a/b/c  (TSM-MBV2 trunc, sequence mode, per-frame KD)
#   3..5 -> C2a/b/c  (Stacked MobileNet, stacked mode, video-level KD)
#   6..8 -> C3a/b/c  (Hybrid: spatial early + TSM late, sequence, per-frame KD)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --array=0-8%9
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/path_c_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/path_c_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

VARIANTS=("C1a" "C1b" "C1c" "C2a" "C2b" "C2c" "C3a" "C3b" "C3c")
V="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"

echo "== node info =="; hostname; echo "ARRAY_TASK=${SLURM_ARRAY_TASK_ID}  variant=${V}"
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
