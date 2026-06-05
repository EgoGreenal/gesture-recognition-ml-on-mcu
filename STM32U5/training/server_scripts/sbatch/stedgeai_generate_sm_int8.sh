#!/bin/bash
#SBATCH --job-name=stedge_sm
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# B (rerun): stedgeai generate on the REAL INT8 streaming TFLite produced by
# the SavedModel + dict-calibrator workaround. Previous run (41517300) raced
# with the SM INT8 export and picked up the FP32 fallback instead.
#   0 -> C1f real INT8
#   1 -> C1j real INT8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --exclude=lrdn1508,lrdn3440
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/stedge_sm_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/stedge_sm_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

TASKS=(
    "C1f models/student_C1F/deploy/C1f_ptq_streaming_sm_int8.tflite models/student_C1F/deploy/stedgeai_generate_int8"
    "C1j models/student_C1J/deploy/C1j_ptq_streaming_sm_int8.tflite models/student_C1J/deploy/stedgeai_generate_int8"
)
SPEC="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
V=$(echo "${SPEC}" | awk '{print $1}')
TFLITE=$(echo "${SPEC}" | awk '{print $2}')
OUT_DIR=$(echo "${SPEC}" | awk '{print $3}')

# Sanity: the file must exist and be INT8 (not FP32 fallback by name)
if [ ! -f "${TFLITE}" ]; then echo "ERROR: ${TFLITE} not found"; exit 1; fi

echo "== node info =="; hostname
echo "variant=${V}  tflite=${TFLITE}"
ls -la "${TFLITE}"

WORK_DIR="${OUT_DIR}/workspace"
NAME="student_${V}_ptq_int8"
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
ls -lR "${OUT_DIR}" | head -30
