#!/bin/bash
#SBATCH --job-name=early_exit
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 12 Tier 1: early-exit S1 + S3 sweep on Top-2 PTQ INT8 clip TFLite.
# Full Jester val (14787) per array task; ~25-35 min wall each on compute CPU.
#
#   0 -> C1f_ptq
#   1 -> C1j_ptq
#
# lat_per_frame_ms is filled in after stedgeai analyze (Day 11 produces it).
# If unknown at submit time, sbatch still produces accuracy + obs_ratio columns;
# latency can be filled in post-hoc using the analyze_int8.py output.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --exclude=lrdn1508
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/early_exit_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/early_exit_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

TASKS=(
    "C1f ptq models/student_C1F/deploy/C1f_ptq_clip_int8.tflite 63.20"
    "C1j ptq models/student_C1J/deploy/C1j_ptq_clip_int8.tflite 32.11"
)
SPEC="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
V=$(echo "${SPEC}" | awk '{print $1}')
TAG=$(echo "${SPEC}" | awk '{print $2}')
TFLITE=$(echo "${SPEC}" | awk '{print $3}')
LAT=$(echo "${SPEC}" | awk '{print $4}')      # ms/frame from Day 10 (FP32 fallback); refine post-Day-11 stedgeai

OUT_JSON="${PROJECT_ROOT}/models/early_exit_${V}_${TAG}.json"

echo "== node info =="; hostname; nproc
echo "task=${SLURM_ARRAY_TASK_ID} V=${V} tag=${TAG} tflite=${TFLITE} lat_per_frame=${LAT}ms"

# Smoke test on login node (50 samples) showed INT8 logits are concentrated -- max
# softmax at frame 0 routinely > 0.85 even when prediction wrong. Default
# Plan-quoted thresh=0.85 / entropy=0.5 exits too early (frame 0-1, ~18% acc).
# Use denser high-confidence grids that span the true acc-vs-obs tradeoff range.
python "${PROJECT_ROOT}/scripts/early_exit_eval.py" \
    --tflite "${TFLITE}" \
    --variant "${V}" \
    --tag "${TAG}" \
    --lat_per_frame_ms "${LAT}" \
    --max_samples 0 \
    --s1_thresholds 0.85 0.90 0.93 0.95 0.97 0.98 0.99 0.995 0.999 \
    --s3_thresholds 0.01 0.05 0.10 0.20 0.30 0.50 0.80 1.20 2.00 \
    --out_json "${OUT_JSON}"
