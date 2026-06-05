#!/bin/bash
#SBATCH --job-name=topk_c1j
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=30G --time=00:20:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/topk_c1j_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/topk_c1j_%j.err
set -euo pipefail
cd /leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU
source scripts/env_setup.sh tf
export PYTHONUNBUFFERED=1
python scripts/eval_topk_c1j.py
