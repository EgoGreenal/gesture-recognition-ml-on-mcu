#!/bin/bash
#SBATCH --job-name=stream_sm
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# A: SavedModel + dict-keyed calibrator workaround for multi-input streaming
# INT8 PTQ. Day 10 documented this workaround but never tested. If it succeeds
# we get real INT8 streaming TFLite for Day 13 deployment without depending on
# X-CUBE-AI 9.1+ to re-do PTQ.
#   0 -> C1f streaming via SavedModel
#   1 -> C1j streaming via SavedModel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --exclude=lrdn1508,lrdn3440
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/stream_sm_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/stream_sm_%A_%a.err

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

echo "== node info =="; hostname

python "${PROJECT_ROOT}/scripts/export_int8.py" \
    --variant "${V}" \
    --ckpt "${CKPT}" \
    --tag ptq \
    --export_streaming \
    --via_saved_model \
    --n_calib 32
