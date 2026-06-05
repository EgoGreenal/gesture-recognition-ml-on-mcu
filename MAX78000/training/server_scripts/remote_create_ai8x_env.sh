#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
AI8X="$PROJECT/ai8x-training"
ENV_DIR="$PROJECT/envs/ai8x-py311"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  mkdir -p "$(dirname "$ENV_DIR")"
  conda create -y -p "$ENV_DIR" python=3.11 pip
fi

conda activate "$ENV_DIR"
python -m pip install "pip==24.2" "setuptools==70.3.0" wheel

cd "$AI8X"
python -m pip install --no-build-isolation -r requirements-base.txt
python -m pip install --no-build-isolation -e distiller --config-settings editable_mode=strict

python - <<'PY'
import torch
import torchvision
import distiller

print("python-ready")
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda", torch.cuda.is_available())
print("distiller", distiller.__version__)
PY
