#!/usr/bin/env bash
set -euo pipefail

DATASET="/path/to/20bn-jester-v1-complete"
PROJECT="/path/to/20bn-jester"

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "pwd=$(pwd)"
echo "conda=$(command -v conda || true)"
echo "dataset=$DATASET"
echo "project=$PROJECT"

cat > "$PROJECT/probe_env.py" <<'PY'
import os
import sys

import torch

print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
    print("cuda_count", torch.cuda.device_count())
PY

conda run -n ai4eda python "$PROJECT/probe_env.py"

echo "dataset root files:"
find "$DATASET" -maxdepth 1 -type f -printf "%f\n" | sort | head -80

echo "dataset root directories:"
find "$DATASET" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort | head -80

echo "metadata candidates:"
find "$DATASET" -maxdepth 2 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.txt" \) -printf "%p\n" | sort | head -80

echo "sample video directories:"
find "$DATASET/20bn-jester-v1" -maxdepth 1 -mindepth 1 -type d | head -5

first_dir=$(find "$DATASET/20bn-jester-v1" -maxdepth 1 -mindepth 1 -type d | head -1)
if [[ -n "${first_dir:-}" ]]; then
  echo "first sample dir=$first_dir"
  find "$first_dir" -maxdepth 1 -type f | sort | head -20
fi

echo "counts:"
find "$DATASET/20bn-jester-v1" -maxdepth 1 -mindepth 1 -type d | wc -l
