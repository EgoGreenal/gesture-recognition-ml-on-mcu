#!/bin/bash
#SBATCH --job-name=export_ta_sl
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
# Day 7 per-frame TA soft labels export. Single GPU + dbg QoS (~30min cap)
# is plenty: 118562 clips × 8 frames × 27 classes -> 52 MB FP16.
# Wall estimate: 8-15 min no-TTA, 15-25 min with TTA.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/export_ta_sl_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/export_ta_sl_%j.err

set -euo pipefail

PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/env_setup.sh" torch

echo "== node info =="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
nvidia-smi
echo

echo "== exporting TA per-frame soft labels (val sanity + train export) =="
python "${PROJECT_ROOT}/scripts/export_ta_softlabels.py" \
    --ckpt "${PROJECT_ROOT}/models/teacher_assistant/best.pt" \
    --out  "${PROJECT_ROOT}/soft_labels/soft_labels_ta_train.npy" \
    --split train \
    --batch_size 64 \
    --num_workers 16 \
    --T 8 \
    --img_size 112 \
    --eval_val
