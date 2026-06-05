#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
cd "$PROJECT"

echo "project=$PROJECT"

if [[ ! -d ai8x-training/.git ]]; then
  git clone https://github.com/analogdevicesinc/ai8x-training.git
else
  git -C ai8x-training rev-parse --short HEAD
fi

mkdir -p annotations

if [[ ! -f annotations/jester-v1-train.csv || ! -f annotations/jester-v1-validation.csv || ! -f annotations/jester-v1-labels.csv ]]; then
  echo "Downloading annotation CSV bundle from Kaggle public dataset kylecloud/jester-csv"
  curl -L --fail --retry 3 \
    -o annotations/jester-csv.zip \
    "https://www.kaggle.com/api/v1/datasets/download/kylecloud/jester-csv"
  python - <<'PY'
import zipfile
from pathlib import Path

root = Path("annotations")
zip_path = root / "jester-csv.zip"
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(root)
print("annotation files:")
for path in sorted(root.glob("*")):
    print(path, path.stat().st_size)
PY
else
  echo "Annotation CSVs already exist."
fi

conda run -n ai4eda python - <<'PY'
import importlib
for name in ["torch", "torchvision", "PIL", "numpy", "tqdm"]:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "ok"))
    except Exception as exc:
        print(name, "MISSING", exc)
PY
