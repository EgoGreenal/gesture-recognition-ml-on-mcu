#!/bin/bash
#SBATCH --job-name=stedge_gen
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# B: stedgeai generate -- not just analyze. Outputs network.c/.h + network_data.c
# + report files, giving us a Day 13 preview of:
#   - actual C code layout
#   - kernel selection (per-layer)
#   - memory map (RAM/ROM regions, weight placement)
#   - any unsupported / fallback ops
# Operates on PTQ streaming TFLite (preferentially INT8 if SavedModel workaround
# succeeded; otherwise FP32 fallback).
#   0 -> C1f streaming
#   1 -> C1j streaming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --exclude=lrdn1508,lrdn3440
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/stedge_gen_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/stedge_gen_%A_%a.err

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

# Prefer real INT8 (from SavedModel workaround if present); else SavedModel FP32 fallback
TFLITE_INT8_SM="${DEPLOY_DIR}/${V}_ptq_streaming_sm_int8.tflite"
TFLITE_INT8_SM_FP32="${DEPLOY_DIR}/${V}_ptq_streaming_sm_int8.fp32.tflite"
TFLITE_INT8="${DEPLOY_DIR}/${V}_ptq_streaming_int8.tflite"
TFLITE_FP32="${DEPLOY_DIR}/${V}_ptq_streaming_int8.fp32.tflite"
if [ -f "${TFLITE_INT8_SM}" ]; then
    TFLITE="${TFLITE_INT8_SM}"; KIND="int8_sm"
elif [ -f "${TFLITE_INT8}" ]; then
    TFLITE="${TFLITE_INT8}"; KIND="int8"
elif [ -f "${TFLITE_INT8_SM_FP32}" ]; then
    TFLITE="${TFLITE_INT8_SM_FP32}"; KIND="fp32_sm"
elif [ -f "${TFLITE_FP32}" ]; then
    TFLITE="${TFLITE_FP32}"; KIND="fp32"
else
    echo "ERROR: no streaming tflite found at ${DEPLOY_DIR}"; ls -la "${DEPLOY_DIR}"; exit 1
fi

echo "== node info =="; hostname
echo "variant=${V}  kind=${KIND}  tflite=${TFLITE}"

OUT_DIR="${DEPLOY_DIR}/stedgeai_generate"
WORK_DIR="${OUT_DIR}/workspace"
NAME="student_${V}_ptq_${KIND}"

mkdir -p "${OUT_DIR}"

stedgeai generate \
    --model "${TFLITE}" \
    --target stm32 \
    --name "${NAME}" \
    --workspace "${WORK_DIR}" \
    --output "${OUT_DIR}" \
    --verbosity 2 \
    2>&1 | tee "${OUT_DIR}/generate_log.txt"

echo
echo "== generated files =="
ls -lR "${OUT_DIR}" | head -60

# Stash the generated network.c/.h source headers (small) under deploy/
GEN_SRC="${OUT_DIR}/${NAME}_data.h"
if [ -f "${GEN_SRC}" ]; then
    head -40 "${GEN_SRC}" || true
fi
