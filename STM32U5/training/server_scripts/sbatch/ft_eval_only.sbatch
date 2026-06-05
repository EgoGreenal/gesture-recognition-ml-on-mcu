#!/bin/bash
#SBATCH --job-name=ft_eval_only
#SBATCH --account=IscrB_WearUsFM
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/ft_eval_%j.out
#SBATCH --error=/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/logs/ft_eval_%j.err
#
# Re-run SWA+TTA eval after the previous job's eval phase crashed (best.pt missing in ft dir).
# Single GPU, ~15-30 min wall.

set -euo pipefail
PROJECT_ROOT="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/env_setup.sh" torch

python "${PROJECT_ROOT}/scripts/swa_tta_eval.py" \
    --ckpt_dir "${PROJECT_ROOT}/models/super_teacher_ft" \
    --epochs_for_swa 1 2 3 4 5 \
    --log_path "${PROJECT_ROOT}/logs/ft_swa_eval.json"

echo
echo "== summary =="
python3 -c "
import json
with open('${PROJECT_ROOT}/logs/ft_swa_eval.json') as f:
    d = json.load(f)
print('config           top1     top5')
best_k = max(d, key=lambda k: d[k]['top1'])
for k, v in d.items():
    star = ' ★' if k == best_k else ''
    flag = ' ≥0.951' if v['top1'] >= 0.951 else (' ≥0.950' if v['top1'] >= 0.950 else '')
    print(f'  {k:18s} {v[\"top1\"]:.4f}  {v[\"top5\"]:.4f}{star}{flag}')
print(f'\\nbest = {best_k} @ {d[best_k][\"top1\"]:.4f}')
"
