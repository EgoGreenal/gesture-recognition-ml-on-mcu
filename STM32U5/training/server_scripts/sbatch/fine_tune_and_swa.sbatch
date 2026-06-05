#!/bin/bash
#SBATCH --job-name=ft_swa
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/ft_swa_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/ft_swa_%j.err
#
# Day 5 path-B: fine-tune 5 epoch (constant lr=2e-5) + SWA over all 5 snapshots + TTA eval.
# 2 GPU DDP; expect ~15 min ft + ~5 min swa+tta eval = ~20-30 min wall.

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" torch

echo "== fine-tune (DDP 2 GPU) =="
torchrun --standalone --nproc_per_node=2 --max_restarts=0 \
    "${PROJECT_ROOT}/scripts/fine_tune.py" \
    --epochs 5 --lr 2e-5 --batch_size 16 --T 8 --img_size 112 \
    --num_workers 6 \
    --job_id "${SLURM_JOB_ID:-local}"

echo
echo "== SWA over fine-tune snapshots + TTA eval (single GPU) =="
# Note: swa_tta_eval.py only uses one GPU. We have 2 allocated but waste 1 here; OK for ~5min.
CUDA_VISIBLE_DEVICES=0 python "${PROJECT_ROOT}/scripts/swa_tta_eval.py" \
    --ckpt_dir "${PROJECT_ROOT}/models/super_teacher_ft" \
    --epochs_for_swa 1 2 3 4 5 \
    --log_path "${PROJECT_ROOT}/logs/ft_swa_eval.json"

echo
echo "== Pretty summary =="
python3 -c "
import json
with open('${PROJECT_ROOT}/logs/ft_swa_eval.json') as f:
    d = json.load(f)
print('config           top1     top5')
for k, v in d.items():
    print(f'  {k:14s} {v[\"top1\"]:.4f}  {v[\"top5\"]:.4f}')
"
