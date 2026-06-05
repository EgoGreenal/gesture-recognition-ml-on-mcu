#!/bin/bash
#SBATCH --job-name=analyze_path_c
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
# Day 10 Path C analyze: 9 variants × (INT8 export + stedgeai analyze).
# Per Day 8 probe: ~20s/variant stedgeai + ~30s INT8 calib rollout. Total ~10
# min. GPU helps eager rollouts for calibration.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/analyze_path_c_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/analyze_path_c_%j.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

echo "== node info =="; hostname; which stedgeai; stedgeai --version 2>&1 | head -2; echo

python "${PROJECT_ROOT}/scripts/analyze_path_c.py" \
    --sram_budget_kb 600 \
    --lat_budget_ms_per_frame 150
