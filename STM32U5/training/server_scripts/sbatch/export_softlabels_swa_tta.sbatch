#!/bin/bash
#SBATCH --job-name=softlabels_swa_tta
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/softlabels_swa_tta_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/softlabels_swa_tta_%j.err
#
# Day 5 path-B final: export soft labels from SWA(fine-tune 5 ckpt) ckpt with TTA enabled.
# Reaches 95.04% val_top1, surpasses original best.pt 94.71%.
# TTA doubles forward passes → ~60 min wall on single A100 (vs 30 min no-TTA).

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" torch

python "${PROJECT_ROOT}/scripts/export_softlabels.py" \
    --ckpt "${PROJECT_ROOT}/models/super_teacher_ft/swa.pt" \
    --out "${PROJECT_ROOT}/soft_labels/soft_labels_train_swa_tta.npy" \
    --split train \
    --batch_size 64 \
    --num_workers 8 \
    --T 8 \
    --img_size 112 \
    --tta

echo
echo "== verify .npy =="
python3 -c "
import numpy as np
sl = np.load('${PROJECT_ROOT}/soft_labels/soft_labels_train_swa_tta.npy')
print(f'shape={sl.shape}  dtype={sl.dtype}  size={sl.nbytes/1e6:.2f}MB')
"
