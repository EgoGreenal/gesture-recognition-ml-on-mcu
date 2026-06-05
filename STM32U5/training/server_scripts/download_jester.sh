#!/bin/bash
# Download 20BN-Jester V1 (~22 GB) from Kaggle into /leonardo_work data dir.
#
# Prereq: ~/.kaggle/kaggle.json present (chmod 600). Get it from
#   https://www.kaggle.com/settings -> "Create New API Token".
#
# Run from a login node (has internet). Background-friendly:
#   nohup bash scripts/download_jester.sh > logs/download_jester.log 2>&1 &
#
# The Kaggle dataset is split into 23 parts (20bn-jester-v1-00..22). The CLI
# fetches them all into one directory. After download we unzip in place.

set -euo pipefail

PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
DATA_DIR="${PROJECT_ROOT}/data/jester"
RAW_DIR="${DATA_DIR}/_raw"

# Need kaggle CLI -> use torch_env (kaggle is in its requirements)
module purge >/dev/null 2>&1 || true
module load profile/deeplrn
module load python/3.11.7
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/venvs/torch_env/bin/activate"

if [ -n "${KAGGLE_API_TOKEN:-}" ]; then
  echo "[download_jester] Using KAGGLE_API_TOKEN env var (new-style token)."
elif [ -f "${HOME}/.kaggle/kaggle.json" ]; then
  chmod 600 "${HOME}/.kaggle/kaggle.json"
  echo "[download_jester] Using ${HOME}/.kaggle/kaggle.json"
else
  echo "ERROR: no Kaggle credentials found."
  echo "  Either set KAGGLE_API_TOKEN=KGAT_... in env,"
  echo "  or place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)."
  exit 1
fi

mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

echo "[download_jester] Dataset: toxicmender/20bn-jester  -> ${RAW_DIR}"
echo "[download_jester] Start time: $(date)"
echo "[download_jester] Free space before: $(df -h "${RAW_DIR}" | tail -n1)"

# --unzip flag auto-extracts each part. Resumes if interrupted (kaggle CLI checks file size).
kaggle datasets download -d toxicmender/20bn-jester -p "${RAW_DIR}" --unzip

echo "[download_jester] End time: $(date)"
echo "[download_jester] Free space after: $(df -h "${RAW_DIR}" | tail -n1)"
echo "[download_jester] Contents:"
ls -lh "${RAW_DIR}" | head -n 20

echo "[download_jester] Done. Verify with: python scripts/verify_jester.py"
