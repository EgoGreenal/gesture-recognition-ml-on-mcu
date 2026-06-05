#!/bin/bash
#SBATCH --job-name=ee_c1f_re
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
# Re-run of 41508683_0 (C1f early_exit) -- original task hit node lrdn3440 with
# corrupted numpy on Lustre cache. Single task, exclude both known-bad nodes.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --exclude=lrdn1508,lrdn3440
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/ee_c1f_re_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/ee_c1f_re_%j.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

echo "== node info =="; hostname

python "${PROJECT_ROOT}/scripts/early_exit_eval.py" \
    --tflite "${PROJECT_ROOT}/models/student_C1F/deploy/C1f_ptq_clip_int8.tflite" \
    --variant C1f \
    --tag ptq \
    --lat_per_frame_ms 63.20 \
    --max_samples 0 \
    --s1_thresholds 0.85 0.90 0.93 0.95 0.97 0.98 0.99 0.995 0.999 \
    --s3_thresholds 0.01 0.05 0.10 0.20 0.30 0.50 0.80 1.20 2.00 \
    --out_json "${PROJECT_ROOT}/models/early_exit_C1f_ptq.json"
