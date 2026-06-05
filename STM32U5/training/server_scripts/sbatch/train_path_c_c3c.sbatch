#!/bin/bash
#SBATCH --job-name=path_c_c3c
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
# normal QoS (no --qos line, partition default = normal, prio 40, no
# MaxNodePerAccount cap). Day 10: C3c got bumped out of the 9-variant lprod
# array because lprod has MaxNodePerAccount=8.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/path_c_c3c_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/path_c_c3c_%j.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

echo "== node info =="; hostname; nvidia-smi; echo

python "${PROJECT_ROOT}/scripts/train_student.py" \
    --variant C3c \
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
    --val_every 5 \
    --seed 42 \
    --log_every 200 \
    --job_id "c3c_${SLURM_JOB_ID}"
