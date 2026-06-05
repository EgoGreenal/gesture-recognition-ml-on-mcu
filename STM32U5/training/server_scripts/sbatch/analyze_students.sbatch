#!/bin/bash
#SBATCH --job-name=analyze_students
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
# Day 10: per-variant INT8 streaming export + stedgeai analyze + Pareto filter.
# stedgeai runs on CPU (~20s/variant per Day 8 probe); INT8 export does eager
# rollouts for calibration data (~30s/variant). Total ~6 min for 7 variants.
# 1 GPU is needed only for loading checkpoint + running calibration rollouts
# (eager forward, very small models, GPU is optional but speeds it up).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/analyze_students_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/analyze_students_%j.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

echo "== node info =="; hostname; which stedgeai; stedgeai --version 2>&1 | head -2; echo

python "${PROJECT_ROOT}/scripts/analyze_students.py" \
    --sram_budget_kb 600 \
    --latency_budget_ms 200
