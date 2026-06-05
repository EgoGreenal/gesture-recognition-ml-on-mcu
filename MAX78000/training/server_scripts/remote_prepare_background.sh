#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/20bn-jester"
LOG="$PROJECT/logs/prepare_resources.log"
mkdir -p "$PROJECT/logs" "$PROJECT/annotations"

cd "$PROJECT"

echo "[$(date)] starting prepare_resources" | tee "$LOG"

echo "[$(date)] stopping previous prepare/git clone processes from this project if any" | tee -a "$LOG"
pkill -f "$PROJECT/remote_prepare_resources.sh" 2>/dev/null || true
pkill -f "git clone https://github.com/analogdevicesinc/ai8x-training.git" 2>/dev/null || true
pkill -f "git-remote-https origin https://github.com/analogdevicesinc/ai8x-training.git" 2>/dev/null || true
pkill -f "git index-pack.*ai8x-training" 2>/dev/null || true
sleep 2

if [[ ! -d ai8x-training/.git ]] || ! git -C ai8x-training rev-parse --short HEAD >/dev/null 2>&1; then
  echo "[$(date)] removing incomplete ai8x-training checkout" | tee -a "$LOG"
  rm -rf ai8x-training
  echo "[$(date)] downloading ai8x-training develop tarball" | tee -a "$LOG"
  curl -L --fail --retry 3 \
    -o ai8x-training-develop.tar.gz \
    "https://github.com/analogdevicesinc/ai8x-training/archive/refs/heads/develop.tar.gz" 2>&1 | tee -a "$LOG"
  mkdir -p ai8x-training
  tar -xzf ai8x-training-develop.tar.gz --strip-components=1 -C ai8x-training
  echo "[$(date)] ai8x-training tarball extracted" | tee -a "$LOG"
else
  echo "[$(date)] ai8x-training already valid: $(git -C ai8x-training rev-parse --short HEAD)" | tee -a "$LOG"
fi

if [[ ! -f annotations/jester-v1-train.csv || ! -f annotations/jester-v1-validation.csv || ! -f annotations/jester-v1-labels.csv ]]; then
  echo "[$(date)] downloading Jester annotation CSVs from Kaggle public dataset kylecloud/jester-csv" | tee -a "$LOG"
  curl -L --fail --retry 3 \
    -o annotations/jester-csv.zip \
    "https://www.kaggle.com/api/v1/datasets/download/kylecloud/jester-csv" 2>&1 | tee -a "$LOG"
  python - <<'PY' 2>&1 | tee -a "$LOG"
import zipfile
from pathlib import Path

root = Path("annotations")
with zipfile.ZipFile(root / "jester-csv.zip") as zf:
    zf.extractall(root)
for path in sorted(root.glob("*")):
    print(path, path.stat().st_size)
PY
else
  echo "[$(date)] annotation CSVs already present" | tee -a "$LOG"
fi

echo "[$(date)] checking Python packages" | tee -a "$LOG"
conda run -n ai4eda python - <<'PY' 2>&1 | tee -a "$LOG"
import importlib
for name in ["torch", "torchvision", "PIL", "numpy", "tqdm"]:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "ok"))
    except Exception as exc:
        print(name, "MISSING", exc)
PY

echo "[$(date)] prepare_resources complete" | tee -a "$LOG"
