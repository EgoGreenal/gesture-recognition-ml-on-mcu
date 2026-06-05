# Training Sources (STM32U5)

This directory contains the project-owned training, distillation, and evaluation
pipeline for the STM32U5 deployment track (TAKD + QAT + Path-C student search).

- `core/`: main Python pipeline — data loaders (HDF5-backed), Teacher / TA /
  student model definitions, KD losses, training, fine-tune, QAT, INT8 export,
  TSM streaming-layer probe, and evaluation / analysis scripts. Framework split:
  PyTorch for Teacher/TA, TensorFlow/Keras for students (bridged via `.npy`
  soft labels). `requirements_torch.txt` / `requirements_tf.txt` pin the two envs.
- `server_scripts/`: shell wrappers used on the HPC (Leonardo) training server —
  env setup, dependency install, dataset download, and `sbatch/` SLURM job
  scripts (all jobs use `--account=IscrB_WearUsFM`).

External repositories (MIT TSM, Video Swin), the 20BN-Jester dataset, the X-CUBE-AI /
ST Edge AI Core toolchain, and large ML artifacts (`*.h5`, `*.npy`, `*.pt`,
`*.tflite`) are not included. See `../docs/WORKFLOW.md`.
