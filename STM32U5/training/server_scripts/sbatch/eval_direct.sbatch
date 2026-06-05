#!/bin/bash
#SBATCH --job-name=eval_direct
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
# Day 10 sync eval (no tf.data, no threading). 1 GPU for clip_model forward;
# CPU side does PIL decode in a serial for-loop. Per Day 9 lesson: this is
# slower than parallel tf.data but reliably non-deadlocking when many other
# SLURM tasks are concurrently reading the same HDF5 file.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/eval_direct_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/eval_direct_%j.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

echo "== node info =="; hostname; nvidia-smi; echo

VARIANTS="${VARIANTS:-V0 V1 V2 V3 V4 V5 V6}"

python "${PROJECT_ROOT}/scripts/eval_direct.py" \
    --variants ${VARIANTS} \
    --batch_size 64 \
    --prefer last
