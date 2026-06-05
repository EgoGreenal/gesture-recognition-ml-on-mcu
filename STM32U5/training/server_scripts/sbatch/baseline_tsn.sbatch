#!/bin/bash
#SBATCH --job-name=baseline_tsn
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/baseline_tsn_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/baseline_tsn_%j.err
#
# Day 2 baseline: 1 epoch ResNet-18 TSN on Jester Pipeline H (112x112 RGB).
# Accept if val top-1 > 55%.
# Time budget: 2h walltime should be ample (estimate 30-60 min for 1 epoch + val).

set -euo pipefail

PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/env_setup.sh" torch

echo "== node info =="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
nvidia-smi || true
echo

echo "== baseline_tsn.py =="
python "${PROJECT_ROOT}/scripts/baseline_tsn.py" \
    --batch_size 64 \
    --num_workers 8 \
    --lr 1e-3 \
    --epochs 1 \
    --T 8 \
    --img_size 112 \
    --job_id "${SLURM_JOB_ID:-local}"
