# Training Sources

This directory contains project-owned training and experiment code.

- `core/`: main Python training, distillation, evaluation, and parsing scripts.
- `server_scripts/`: shell wrappers used on the Linux training server.
- `ai8x_custom_files/`: custom dataset adapters, ai8x models, QAT policies, and ai8x-synthesis network YAML files.
- `experiments/`: early-exit, 3D CNN, and reporting experiments.

External ai8x repositories and datasets are not included. See `../docs/INSTALL.md` and `../docs/SERVER_TRAINING.md`.

The compact all-model training summary is in `../docs/TRAINING_HISTORY_SUMMARY.md`.
