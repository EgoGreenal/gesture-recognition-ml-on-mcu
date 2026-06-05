#!/bin/bash
# Build the torch_env venv: PyTorch 2.4 cu121 + training stack.
# Run on a login node (has internet) or via srun on compute node.
# Idempotent: re-running upgrades packages but does not destroy the venv.

set -euo pipefail

PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
VENV="${PROJECT_ROOT}/venvs/torch_env"

module purge >/dev/null 2>&1 || true
module load profile/deeplrn
module load python/3.11.7
module load cuda/12.2
module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2

mkdir -p "${PROJECT_ROOT}/cache/pip"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"

if [ ! -d "${VENV}" ]; then
  echo "[install_torch_env] Creating venv: ${VENV}"
  python -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r "${PROJECT_ROOT}/scripts/requirements_torch.txt"

echo "[install_torch_env] Done. Verify with: python scripts/check_torch.py"
