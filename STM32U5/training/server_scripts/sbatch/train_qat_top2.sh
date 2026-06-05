#!/bin/bash
#SBATCH --job-name=qat_top2
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 11: QAT (fake-quant STE) fine-tune for Top-2 (C1f + C1j) starting from
# Day 10 FP32 ckpts. 5 epochs × bs=64 × lr=1e-5 × KD (TA per-frame soft labels).
#
#   0 -> C1f  (1.39M, 87.35% FP32)  ckpt: student_C1F/41478885_2/best.weights.h5
#   1 -> C1j  (722K,  86.65% FP32)  ckpt: student_C1J/41478885_6/best.weights.h5
#
# Each task: 1 GPU, ~30-60 min, ~1.5 GPU-hr total budget for Day 11.
#
# Day 10 lesson: --exclude=lrdn1508 (had hang-21min incident with two tasks on
# the same node). lprod QoS (12h cap, prio 40) -- normal/dbg have submit limit
# and dbg 25min cap is too short for 5-epoch.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --exclude=lrdn1508
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/qat_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/qat_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

TASKS=(
    "C1f models/student_C1F/41478885_2/best.weights.h5"
    "C1j models/student_C1J/41478885_6/best.weights.h5"
)
SPEC="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
V=$(echo "${SPEC}" | awk '{print $1}')
CKPT=$(echo "${SPEC}" | awk '{print $2}')

echo "== node info =="; hostname; echo "task=${SLURM_ARRAY_TASK_ID} variant=${V} ckpt=${CKPT}"
nvidia-smi; echo

JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

python "${PROJECT_ROOT}/scripts/train_qat.py" \
    --variant "${V}" \
    --load_fp32_weights "${CKPT}" \
    --soft_labels_ta "${PROJECT_ROOT}/soft_labels/soft_labels_ta_train.npy" \
    --epochs 5 \
    --warmup_epochs 0 \
    --batch_size 64 \
    --lr 1e-5 \
    --wd 1e-4 \
    --alpha 0.3 \
    --temperature 4.0 \
    --frame_weights 0.10 0.20 0.30 0.40 \
    --num_workers 8 \
    --fast_val_batches 30 \
    --val_every 2 \
    --seed 42 \
    --log_every 200 \
    --job_id "${JOB_ID}"
