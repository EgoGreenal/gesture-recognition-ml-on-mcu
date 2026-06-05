#!/bin/bash
#SBATCH --job-name=analyze_int8
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 11: stedgeai analyze on PTQ streaming-form INT8 TFLite for Top-2.
# Uses scripts/sbatch/export_streaming_int8.sh outputs (clip OR fp32-fallback).
# stedgeai is CPU-only, ~30s-2min per analyze. 1 task array element per variant.
#   0 -> C1f streaming
#   1 -> C1j streaming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --exclude=lrdn1508
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/analyze_int8_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/analyze_int8_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

TASKS=(
    "C1f models/student_C1F/deploy"
    "C1j models/student_C1J/deploy"
)
SPEC="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
V=$(echo "${SPEC}" | awk '{print $1}')
DEPLOY_DIR=$(echo "${SPEC}" | awk '{print $2}')

# Pick INT8 streaming tflite preferentially; fall back to FP32 if INT8 export failed
TFLITE_INT8="${DEPLOY_DIR}/${V}_ptq_streaming_int8.tflite"
TFLITE_FP32="${DEPLOY_DIR}/${V}_ptq_streaming_int8.fp32.tflite"
if [ -f "${TFLITE_INT8}" ]; then
    TFLITE="${TFLITE_INT8}"
    KIND="int8"
elif [ -f "${TFLITE_FP32}" ]; then
    TFLITE="${TFLITE_FP32}"
    KIND="fp32"
else
    echo "ERROR: no streaming tflite found at ${DEPLOY_DIR}"
    ls -la "${DEPLOY_DIR}"
    exit 1
fi

echo "== node info =="; hostname
echo "variant=${V}  kind=${KIND}  tflite=${TFLITE}"

OUT_DIR="${DEPLOY_DIR}/stedgeai_day11_ptq"
OUT_JSON="${PROJECT_ROOT}/models/stedgeai_${V}_ptq.json"

python "${PROJECT_ROOT}/scripts/analyze_int8.py" \
    --tflite "${TFLITE}" \
    --name "student_${V}_ptq" \
    --out_dir "${OUT_DIR}" \
    --out_json "${OUT_JSON}" \
    --T 8
