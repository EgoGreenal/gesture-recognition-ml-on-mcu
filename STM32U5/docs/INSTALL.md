# INSTALL — STM32U5 Track Environment

This track was developed on the **Leonardo HPC cluster** (SLURM, no Docker —
Singularity-only security policy). The environment is therefore built with
`module load` + Python virtualenvs rather than containers.

## 1. Toolchain overview

| Component | Version | Purpose |
|---|---|---|
| Python | 3.11.7 (`module load python/3.11.7`) | base interpreter |
| CUDA / cuDNN | 12.2 / 8.9.7.29 | GPU training |
| PyTorch | 2.4 (cu121) — `torch_env` | Super Teacher + TA training |
| TensorFlow + tfmot | 2.15 + tfmot — `tf_env` | Student training, QAT, INT8 TFLite export |
| ST Edge AI Core | 4.0 (= STM32Cube.AI 12.0) | model analyze / C-code generation |
| ARM GNU toolchain | 13.2.Rel1 (`arm-none-eabi-gcc`) | U5 firmware build |

**Two separate venvs** are used on purpose — PyTorch (cuDNN 9.x wheels) and
TF 2.15 (cuDNN 8.9) conflict in one env. The two stages communicate only through
`.npy` soft-label files, so no model-weight conversion is needed.

> venvs live under `/leonardo_work/.../venvs/` (NOT `$HOME`, which is ~14 GB).

## 2. One-time install

Run on a login node (has internet):

```bash
bash scripts/install_torch_env.sh   # creates venvs/torch_env (PyTorch 2.4 + training stack)
bash scripts/install_tf_env.sh      # creates venvs/tf_env (TF 2.15 + tfmot)
```

Both scripts are idempotent (re-running upgrades packages, never destroys the venv).
Pinned dependency lists: `training/core/requirements_torch.txt`,
`training/core/requirements_tf.txt`.

The ST Edge AI Core (`tools/stedgeai/4.0/...`) and ARM toolchain
(`tools/arm-gnu-toolchain-13.2.Rel1-...`) are vendor downloads, not pip packages.

## 3. Per-session activation

```bash
source scripts/env_setup.sh torch   # Stages 1-2: Teacher + TA (PyTorch)
source scripts/env_setup.sh tf      # Stages 3-4: Student + QAT (TensorFlow)
```

`env_setup.sh` loads the modules, activates the venv, sets `PROJECT_ROOT` /
`DATA_ROOT` / cache dirs (redirected into `/leonardo_work` to avoid filling
`$HOME`), and appends ST Edge AI + ARM toolchain to `PATH`. It is safe to source
repeatedly and never uses `set -e` (would kill an interactive shell).

> **ST Edge AI PATH note:** the install dir ships a bundled `python` (3.9.13);
> `env_setup.sh` *appends* (not prepends) it to `PATH` so it cannot shadow the
> active venv.

## 4. SLURM

All batch jobs **must** set `--account=IscrB_WearUsFM` (the default account is a
different project and would be billed wrongly). Wrappers in
`training/server_scripts/` already do this. See [SERVER_TRAINING.md](SERVER_TRAINING.md).

## 5. Dataset

20BN-Jester V1 (27 classes, 148k clips). The raw tgz expands to 2.4M small JPGs
(Lustre metadata blowup), so it is streamed into a single ~28 GB HDF5 file
(`data/jester/jester_full.h5`) via `training/core/tgz_to_hdf5.py`. See
[TRAINING.md](TRAINING.md) §1.
