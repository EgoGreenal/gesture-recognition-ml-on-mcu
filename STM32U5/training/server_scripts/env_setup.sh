# MLonMCU project environment setup
# Usage:
#   source scripts/env_setup.sh torch   # Days 3-7: Teacher + TA (PyTorch)
#   source scripts/env_setup.sh tf      # Days 8-11: Student + QAT (TensorFlow)
#
# Loads modules + activates the named venv. Safe to source repeatedly.
# Do NOT use `set -e` — would kill the user's shell on any non-zero exit.

PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
export PROJECT_ROOT
export DATA_ROOT="${PROJECT_ROOT}/data"
export MODELS_ROOT="${PROJECT_ROOT}/models"
export LOGS_ROOT="${PROJECT_ROOT}/logs"
export CACHE_ROOT="${PROJECT_ROOT}/cache"
export EXTERNAL_ROOT="${PROJECT_ROOT}/external"
export SLURM_ACCOUNT="IscrB_WearUsFM"

# Module loads
module purge >/dev/null 2>&1
module load profile/deeplrn >/dev/null 2>&1
module load python/3.11.7 >/dev/null 2>&1
module load cuda/12.2 >/dev/null 2>&1
module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2 >/dev/null 2>&1

# Venv selection
WHICH="${1:-torch}"
case "${WHICH}" in
  torch) VENV_PATH="${PROJECT_ROOT}/venvs/torch_env" ;;
  tf)    VENV_PATH="${PROJECT_ROOT}/venvs/tf_env" ;;
  *)
    echo "[env_setup] Unknown env '${WHICH}'. Use 'torch' or 'tf'."
    return 1 2>/dev/null || exit 1
    ;;
esac

if [ -d "${VENV_PATH}" ]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
  echo "[env_setup] Activated venv: ${VENV_PATH}"
else
  echo "[env_setup] Venv not found: ${VENV_PATH}"
  echo "[env_setup] Run: bash ${PROJECT_ROOT}/scripts/install_${WHICH}_env.sh"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ST Edge AI Core 4.0 (= STM32CubeAI 12.0; Day 8 probe + Day 13 deployment).
# Append (NOT prepend) — the install dir also contains a bundled `python` binary
# (Python 3.9.13) which would shadow the active venv if it came first on PATH.
STEDGEAI_ROOT="${PROJECT_ROOT}/tools/stedgeai/4.0/Utilities/linux"
if [ -x "${STEDGEAI_ROOT}/stedgeai" ]; then
  export PATH="${PATH}:${STEDGEAI_ROOT}"
  export STEDGEAI_ROOT
fi

# ARM GNU Embedded Toolchain (arm-none-eabi-gcc 13.2.1; Day 13 U5 burn).
ARM_TOOLCHAIN_ROOT="${PROJECT_ROOT}/tools/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi"
if [ -x "${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-gcc" ]; then
  export PATH="${PATH}:${ARM_TOOLCHAIN_ROOT}/bin"
  export ARM_TOOLCHAIN_ROOT
fi

# Cache dirs in /leonardo_work (avoid filling $HOME)
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TFHUB_CACHE_DIR="${CACHE_ROOT}/tfhub"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export KAGGLE_CONFIG_DIR="${HOME}/.kaggle"

CUDA_HOME=$(dirname "$(dirname "$(which nvcc 2>/dev/null)")")
export CUDA_HOME

echo "[env_setup] which=${WHICH}  python=$(which python) ($(python --version 2>&1))"
echo "[env_setup] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[env_setup] SLURM_ACCOUNT=${SLURM_ACCOUNT}"
