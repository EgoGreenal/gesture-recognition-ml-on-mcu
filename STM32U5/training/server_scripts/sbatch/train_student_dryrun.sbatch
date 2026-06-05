#!/bin/bash
#SBATCH --job-name=train_student_dry
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
# dbg QoS, 30min cap, priority 80 (Day 6/7 lessons: starts in seconds).
# Just enough to verify train_step + val_step + checkpoint save all execute
# on GPU without crash, on the smallest variant (V0).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/student_dry_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/student_dry_%j.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1   # flush stdout/err to SLURM log immediately

echo "== node info =="; hostname; nvidia-smi; echo

python "${PROJECT_ROOT}/scripts/train_student.py" \
    --variant V0 \
    --soft_labels_ta "${PROJECT_ROOT}/soft_labels/soft_labels_ta_train.npy" \
    --epochs 1 \
    --warmup_epochs 0 \
    --batch_size 64 \
    --lr 1e-3 \
    --wd 1e-4 \
    --alpha 0.3 \
    --temperature 4.0 \
    --num_workers 8 \
    --fast_val_batches 5 \
    --val_every 1 \
    --max_steps 15 \
    --log_every 5 \
    --job_id "dry_${SLURM_JOB_ID:-local}"
