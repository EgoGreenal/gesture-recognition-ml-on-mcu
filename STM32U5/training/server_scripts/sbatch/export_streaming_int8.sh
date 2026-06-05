#!/bin/bash
#SBATCH --job-name=stream_int8
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 11: try streaming-form INT8 PTQ export for Top-2 (for stedgeai deployment
# analysis + Day 13 X-CUBE-AI). Day 10 lesson: multi-input INT8 PTQ frequently
# fails (TFLite Converter reorders inputs); export_int8.py falls back to FP32
# TFLite. Either way we get something stedgeai can analyze.
#   0 -> C1f streaming
#   1 -> C1j streaming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --exclude=lrdn1508
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/stream_int8_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/stream_int8_%A_%a.err

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
    --n_calib 32
