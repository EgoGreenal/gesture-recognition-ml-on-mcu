#!/bin/bash
#SBATCH --job-name=eval_int8
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
# Day 11: full Jester val (14787 samples) INT8 inference on PTQ TFLite for Top-2.
#   0 -> C1f_ptq clip INT8 TFLite
#   1 -> C1j_ptq clip INT8 TFLite
# dbg QoS: 25min cap; throughput ~10 samp/s on compute CPU -> ~25min for 14787.
# CPU-only (TFLite Interpreter is CPU; no GPU needed -- save GPU-hr).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:25:00
#SBATCH --array=0-1%2
#SBATCH --exclude=lrdn1508
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/eval_int8_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/eval_int8_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

TASKS=(
    "C1f ptq models/student_C1F/deploy/C1f_ptq_clip_int8.tflite"
    "C1j ptq models/student_C1J/deploy/C1j_ptq_clip_int8.tflite"
)
SPEC="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
V=$(echo "${SPEC}" | awk '{print $1}')
TAG=$(echo "${SPEC}" | awk '{print $2}')
TFLITE=$(echo "${SPEC}" | awk '{print $3}')

# Write per-(variant,tag) JSON to avoid the race condition we hit on login node
OUT_JSON="${PROJECT_ROOT}/models/eval_int8_${V}_${TAG}.json"

echo "== node info =="; hostname; nproc; free -g

python "${PROJECT_ROOT}/scripts/eval_int8.py" \
    --tflite "${TFLITE}" \
    --variant "${V}" \
    --tag "${TAG}" \
    --max_samples 0 \
    --out_json "${OUT_JSON}"
