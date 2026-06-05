#!/bin/bash
#SBATCH --job-name=train_student_v2
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 10 production run #2 (Day 9 array hit ~16% acc with insufficient training).
# Changes from v1:
#   * Data loader: tf.io.decode_jpeg in graph (Day 10 rewrite) -> 5-10x faster
#   * Epochs: 8 -> 30  (Plan original target)
#   * max_steps: 700 -> 0 (no cap; full 1853 steps/epoch at bs=64)
#   * KD frame_weights: [0.25]*4 -> [0.10, 0.20, 0.30, 0.40] (de-weight noisy
#     obs=0.25 supervision; TA frame[1] only 10.7% accurate)
#   * Per-epoch val re-enabled (val_every=3) -- new pipeline isn't GIL-bound
#   * QoS lprod (96h max) instead of normal usr_prod (24h)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-6%7
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/student_v2_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/student_v2_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

VARIANTS=(V0 V1 V2 V3 V4 V5 V6)
V="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"

echo "== node info =="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  variant=${V}"
nvidia-smi
echo

JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

python "${PROJECT_ROOT}/scripts/train_student.py" \
    --variant "${V}" \
    --soft_labels_ta "${PROJECT_ROOT}/soft_labels/soft_labels_ta_train.npy" \
    --epochs 30 \
    --warmup_epochs 2 \
    --batch_size 64 \
    --lr 1e-3 \
    --wd 1e-4 \
    --alpha 0.3 \
    --temperature 4.0 \
    --frame_weights 0.10 0.20 0.30 0.40 \
    --num_workers 8 \
    --max_steps 0 \
    --fast_val_batches 30 \
    --val_every 3 \
    --seed 42 \
    --log_every 200 \
    --job_id "${JOB_ID}"
