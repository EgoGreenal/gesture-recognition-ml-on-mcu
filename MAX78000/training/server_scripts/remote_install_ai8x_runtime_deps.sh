#!/usr/bin/env bash
set -euo pipefail

AI8X="/path/to/20bn-jester/ai8x-training"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai4eda

cd "$AI8X"
python -m pip install torchnet==0.0.4
python -m pip install --no-deps -e distiller --config-settings editable_mode=strict
