# Server Training Scripts

The remote training workflow used shell scripts and tmux sessions on a Linux server with one NVIDIA GPU.

## Layout

- `training/server_scripts/`
  - Top-level launch, setup, monitoring, and QAT scripts copied from the remote project.
- `training/ai8x_custom_files/codex_uploads/`
  - ai8x dataset adapters, model definitions, QAT policies, and install scripts for each model family.
- `training/experiments/`
  - Standalone experiments such as early-exit simulation, 3D CNN exploration, and report generation.

## Original server paths

Many scripts preserve the original server paths:

```bash
PROJECT=/path/to/20bn-jester
DATASET_ROOT=/path/to/20bn-jester-v1-complete
```

If you run on a different server, edit those variables at the top of the scripts.

## Typical workflow

```bash
ssh <user>@<server>
conda activate ai4eda
cd /path/to/20bn-jester
bash remote_start_ai8x_jester_res_signed_distill_deployable_tmux.sh
```

Then monitor the tmux session:

```bash
tmux ls
tmux attach -t jester_res_signed_distill
```

The exact tmux session name is defined in the corresponding `remote_start_*.sh` wrapper.

## Installing custom ai8x files

The `remote_install_*.sh` scripts copy custom project files into an ai8x-training checkout:

- Dataset adapters go to `ai8x-training/datasets/`.
- Model definitions go to `ai8x-training/models/`.
- QAT policy YAML files go to `ai8x-training/policies/`.

The source copies are preserved under `training/ai8x_custom_files/codex_uploads/`.

## Recommended scripts to inspect first

- `remote_run_ai8x_jester_res_signed_distill_deployable.sh`
- `remote_start_ai8x_jester_res_signed_distill_deployable_tmux.sh`
- `remote_run_ai8x_jester_huge_teacher_distill.sh`
- `remote_run_ai8x_jester_huge_teacher_qat_sweep.sh`
- `remote_eval_jester_confusions.sh`

These cover the final deployable student, teacher distillation, QAT sweeps, and validation reporting.
