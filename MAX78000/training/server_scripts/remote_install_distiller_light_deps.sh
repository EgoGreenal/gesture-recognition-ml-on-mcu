#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai4eda

python -m pip install \
  tabulate==0.8.9 \
  pydot==1.4.1 \
  graphviz==0.10.1 \
  pandas \
  matplotlib \
  xlsxwriter \
  scikit-learn \
  pretrainedmodels \
  tqdm
