#!/bin/bash
#SBATCH --job-name=tf_gpu_sanity
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/gpu_sanity_tf_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/gpu_sanity_tf_%j.err
#
# Day 1 leftover: confirm tf_env can see the GPU on a Boost A100 node.
# Expected: check_tf.py reports tensorflow 2.15.1, cuda.is_available=True, 1 GPU.

set -euo pipefail

PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"

# env_setup.sh purges + reloads modules and activates tf_env
source "${PROJECT_ROOT}/scripts/env_setup.sh" tf

echo "== nvidia-smi =="
nvidia-smi || true
echo
echo "== check_tf.py =="
python "${PROJECT_ROOT}/scripts/check_tf.py"
