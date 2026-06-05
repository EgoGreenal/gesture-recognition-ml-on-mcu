#!/bin/bash
#SBATCH --job-name=eval_students
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
# Day 10 post-training val. Each variant: load best.weights.h5, run full
# Jester val (~14787 samples / batch_size 64 = ~231 batches), write
# eval_full.json next to the run. 7 variants × ~5-10 min each = ~45-60 min;
# dbg QoS cap 30 min may NOT be enough -- check first variant time and
# resubmit on normal QoS if needed.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/eval_students_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/eval_students_%j.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

echo "== node info =="; hostname; nvidia-smi; echo

# Optional: pass --variants on `sbatch --export` line for selective eval
VARIANTS="${VARIANTS:-V0 V1 V2 V3 V4 V5 V6}"

python "${PROJECT_ROOT}/scripts/eval_students.py" \
    --variants ${VARIANTS} \
    --batch_size 64 \
    --num_shards 2
