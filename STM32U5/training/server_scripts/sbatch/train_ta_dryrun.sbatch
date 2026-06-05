#!/bin/bash
#SBATCH --job-name=ta_dryrun
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
# QoS dbg: 30min max walltime, higher priority — perfect for a 30-step probe.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/ta_dryrun_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/ta_dryrun_%j.err

set -euo pipefail

PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/env_setup.sh" torch

echo "== node info =="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
nvidia-smi
echo

# Dry-run config:
#   * 1 epoch, only 30 train steps -> ~30s of train + 1 full val pass
#   * Writes ckpt into a scratch dir so it doesn't collide with full-run state
#   * Same DDP/SyncBN/AMP path as the real run
echo "== dry-run: 30 train steps + 1 val pass (2 GPU) =="
torchrun \
    --standalone \
    --nproc_per_node=2 \
    --max_restarts=0 \
    "${PROJECT_ROOT}/scripts/train_ta.py" \
    --epochs 1 \
    --warmup_epochs 0 \
    --batch_size 64 \
    --lr 1e-3 \
    --T 8 \
    --img_size 112 \
    --num_workers 8 \
    --max_train_steps 30 \
    --ckpt_dir "${PROJECT_ROOT}/models/teacher_assistant_dryrun" \
    --job_id "dryrun_${SLURM_JOB_ID:-local}"
