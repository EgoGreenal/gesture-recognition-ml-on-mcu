#!/bin/bash
#SBATCH --job-name=ee_v2
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Day 12 v2 sweep: S1+S3 thresholds x min_exit_frame floors {0, 3, 4, 5}.
# v1 result: vanilla S1/S3 exits too early on TSM C1 architecture (~70% exits
# at frame 0 due to early-frame overconfidence). Adding a minimum-frame floor
# restricts exits to where per-frame acc is reliable.
#
# Per Day 10 metrics: per-frame acc is 18% (f1) -> 46% (f3) -> 86% (f5) -> 87% (f7)
# Floor=4 means earliest exit at obs=0.625 (per-frame acc ~60%)
# Floor=5 means earliest exit at obs=0.750 (per-frame acc ~86%)
#
# Saves collected logits to .npz for future re-runs without TFLite inference.
#   0 -> C1f
#   1 -> C1j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --exclude=lrdn1508,lrdn3440
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/ee_v2_%A_%a.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/ee_v2_%A_%a.err

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
LAT=$(echo "${SPEC}" | awk '{print $4}')

OUT_JSON="${PROJECT_ROOT}/models/early_exit_${V}_${TAG}_v2.json"
LOGITS_NPZ="${PROJECT_ROOT}/models/early_exit_${V}_${TAG}_logits.npz"

echo "== node info =="; hostname; nproc

python "${PROJECT_ROOT}/scripts/early_exit_eval.py" \
    --tflite "${TFLITE}" \
    --variant "${V}" \
    --tag "${TAG}" \
    --lat_per_frame_ms "${LAT}" \
    --max_samples 0 \
    --s1_thresholds 0.85 0.90 0.93 0.95 0.97 0.98 0.99 0.995 0.999 \
    --s3_thresholds 0.01 0.05 0.10 0.20 0.30 0.50 0.80 1.20 2.00 \
    --min_exit_frames 0 3 4 5 \
    --save_logits "${LOGITS_NPZ}" \
    --out_json "${OUT_JSON}"
