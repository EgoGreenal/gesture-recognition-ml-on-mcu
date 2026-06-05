#!/bin/bash
#SBATCH --job-name=analyze_c1d2
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/analyze_c1d2_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/analyze_c1d2_%j.err

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf
export PYTHONUNBUFFERED=1

echo "== node info =="; hostname; which stedgeai; echo

python "${PROJECT_ROOT}/scripts/analyze_path_c.py" \
    --variants C1n C1o C1j \
    --sram_budget_kb 600 \
    --lat_budget_ms_per_frame 150 \
    --out_csv "${PROJECT_ROOT}/models/c1deep2_comparison.csv"
