#!/bin/bash
#SBATCH --job-name=train_ta_4g
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
# Resource profile: 4 GPU full-node (race against the 2-GPU run job 41400167).
# Day 3-5 experience: full-node A100 queue can be ~10h on Leonardo, so this
# may not actually beat the 2-GPU run wall-clock — submitted as a controlled
# experiment per user request.
#   per-GPU batch 64, effective batch 256 (= 2× the 2-GPU run)
#   lr = 5e-4 × world_size=4 = 2e-3 (linear scaling)
#   15 epoch; expected pure-compute wall ~2.5-3.5h IF it dequeues quickly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/train_ta_4gpu_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/train_ta_4gpu_%j.err

set -euo pipefail

PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/env_setup.sh" torch

echo "== node info =="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
nvidia-smi
echo

# Distinct ckpt dir so the 4-GPU run doesn't overwrite the 2-GPU run's state.
# The 2-GPU run keeps the canonical models/teacher_assistant path.
CKPT_DIR="${PROJECT_ROOT}/models/teacher_assistant_4gpu"

echo "== torchrun DDP launch (4 GPU) =="
torchrun \
    --standalone \
    --nproc_per_node=4 \
    --max_restarts=0 \
    "${PROJECT_ROOT}/scripts/train_ta.py" \
    --epochs 15 \
    --warmup_epochs 2 \
    --batch_size 64 \
    --lr 2e-3 \
    --weight_decay 4e-5 \
    --T 8 \
    --img_size 112 \
    --num_workers 8 \
    --alpha 0.5 \
    --temperature 4.0 \
    --n_div 8 \
    --ckpt_every 1 \
    --ckpt_dir "${CKPT_DIR}" \
    --job_id "${SLURM_JOB_ID:-local}"
