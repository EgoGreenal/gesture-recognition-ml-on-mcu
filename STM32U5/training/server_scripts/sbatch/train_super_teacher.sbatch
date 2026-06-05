#!/bin/bash
#SBATCH --job-name=super_teacher
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
# QoS note: account has access to {normal, boost_qos_lprod, boost_qos_bprod, boost_qos_dbg}.
#   - normal (priority 40)   — default, what we use
#   - boost_qos_lprod (40)   — same priority, 4-day walltime (no speedup)
#   - boost_qos_bprod (60)   — higher priority BUT MinTRES=65 nodes (unusable here)
#   - boost_qos_dbg (80)     — higher priority BUT MaxWall=30min (training won't fit)
# So `normal` is the best QoS for this job profile.
#
# Resource profile: 2 GPU (partial node, half of 4-GPU A100 box)
#   * 4-GPU full-node requests sit in queue for many hours on this cluster (1000+ jobs ahead,
#     ~28/hr drain rate observed on 2026-05-11 evening). 2 GPU = half-node share, much
#     easier to backfill into mixed-state nodes that already have 1-2 GPUs in use.
#   * Training time: ~16-24h for 30 epochs (vs 8-12h on 4 GPU); --requeue handles overflow.
#   * Effective batch = 2 * 16 = 32 (vs 64 on 4 GPU); lr scaled linearly to 2e-4.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/super_teacher_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/super_teacher_%j.err

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
    "${PROJECT_ROOT}/scripts/train_super_teacher.py" \
    --epochs 30 \
    --warmup_epochs 3 \
    --batch_size 16 \
    --lr 2e-4 \
    --weight_decay 0.02 \
    --T 8 \
    --img_size 112 \
    --num_workers 6 \
    --drop_path_rate 0.1 \
    --ckpt_every 5 \
    --job_id "${SLURM_JOB_ID:-local}"
