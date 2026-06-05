#!/bin/bash
#SBATCH --job-name=eval_lprod
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Fallback for eval_int8_full_val.sh if dbg cap (25 min) hits. lprod 1h cap.
# Submitted manually only if 41508111 dbg array is killed before completion.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --exclude=lrdn1508,lrdn3440
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/eval_lprod_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/eval_lprod_%A_%a.err

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

OUT_JSON="${PROJECT_ROOT}/models/eval_int8_${V}_${TAG}.json"

echo "== node info =="; hostname

python "${PROJECT_ROOT}/scripts/eval_int8.py" \
    --tflite "${TFLITE}" \
    --variant "${V}" \
    --tag "${TAG}" \
    --max_samples 0 \
    --out_json "${OUT_JSON}"
