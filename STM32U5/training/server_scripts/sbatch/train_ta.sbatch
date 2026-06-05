#!/bin/bash
#SBATCH --job-name=train_ta
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
# Resource profile: 2 GPU partial-node (Day 5 learning: 4-GPU full-node queue
# was ~10h on 2026-05-11; 2-GPU mixed-state backfill ~30min). Plan Day 6
# explicitly switched from 4 GPU to 2 GPU partial-node accordingly.
#   per-GPU batch 64, effective batch 128, lr = 5e-4 × 2 = 1e-3
#   15 epoch, expected wall 4-6h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/train_ta_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/train_ta_%j.err

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

echo "== torchrun DDP launch (2 GPU) =="
torchrun \
    --standalone \
    --nproc_per_node=2 \
    --max_restarts=0 \
    "${PROJECT_ROOT}/scripts/train_ta.py" \
    --epochs 15 \
    --warmup_epochs 2 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 4e-5 \
    --T 8 \
    --img_size 112 \
    --num_workers 8 \
    --alpha 0.5 \
    --temperature 4.0 \
    --n_div 8 \
    --ckpt_every 1 \
    --job_id "${SLURM_JOB_ID:-local}"
