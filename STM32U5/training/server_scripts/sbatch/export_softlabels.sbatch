#!/bin/bash
#SBATCH --job-name=softlabels_export
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/softlabels_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/softlabels_%j.err
#
# Day 5 deliverable: forward Super Teacher over the train split, save per-clip logits.
# Single GPU, ~15-25 min for 118562 clips at batch 64.

set -euo pipefail

PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/env_setup.sh" torch

echo "== node =="
hostname
nvidia-smi -L
echo

python "${PROJECT_ROOT}/scripts/export_softlabels.py" \
    --ckpt "${PROJECT_ROOT}/models/super_teacher/best.pt" \
    --out "${PROJECT_ROOT}/soft_labels/soft_labels_train.npy" \
    --split train \
    --batch_size 64 \
    --num_workers 8 \
    --T 8 \
    --img_size 112
