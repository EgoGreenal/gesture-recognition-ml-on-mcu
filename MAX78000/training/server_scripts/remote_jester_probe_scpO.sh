#!/usr/bin/env bash
set -euo pipefail

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "pwd=$(pwd)"
echo "conda=$(command -v conda || true)"

conda run -n ai4eda python - <<'PY'
import os
import sys

import torch

print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
    print("cuda_count", torch.cuda.device_count())

dataset = "/path/to/20bn-jester-v1-complete"
project = "/path/to/20bn-jester"
print("dataset_exists", os.path.exists(dataset), dataset)
print("project_exists", os.path.exists(project), project)
PY

echo "dataset top-level:"
find /path/to/20bn-jester-v1-complete -maxdepth 2 -mindepth 1 | head -80

echo "project top-level:"
find /path/to/20bn-jester -maxdepth 2 -mindepth 1 | head -80
