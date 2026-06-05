# SERVER_TRAINING — STM32U5 Track (Leonardo HPC / SLURM)

All training and evaluation ran on the **Leonardo** cluster via SLURM batch jobs.
The job wrappers live in `../training/server_scripts/` (setup `.sh`) and
`../training/server_scripts/sbatch/` (SLURM jobs). Environment setup is in
[INSTALL.md](INSTALL.md).

## Golden rule

Every job **must** set `--account=IscrB_WearUsFM`. The default account belongs to a
different project and would be billed wrongly. All committed wrappers already do this.

## Partitions / QoS used

| Setting | Value | Notes |
|---|---|---|
| `--partition` | `boost_usr_prod` | A100 GPU partition |
| `--qos` | `boost_qos_lprod` | long-production (96 h cap), prio 40 |
| `--gres` | `gpu:1` (students) … `gpu:4` (teacher) | see per-stage table in TRAINING.md |
| `--cpus-per-task` | 4–8 | data-loader workers |
| `--mem` | 12–60 G | larger for HDF5-backed loaders |

## Job inventory (`../training/server_scripts/sbatch/`)

- **Teacher / TA**: `train_super_teacher.sbatch` (1×4 GPU DDP), `train_ta.sbatch` /
  `train_ta_4gpu.sbatch`, plus `*_dryrun.*` smoke variants.
- **Student search (Path-C)**: `train_path_c_array.sh` — a 9-task SLURM **array**
  (`--array=0-8`) running C1a/b/c, C2a/b/c, C3a/b/c each on 1 GPU in parallel
  (~1.5–2.5 h each, ~2–3 h wall vs ~18 h sequential). Deeper C1 sweeps:
  `train_c1_deep.sh`, `train_c1_deep2.sh`, `train_c1k_c1m.sh`.
- **Soft-label export**: `export_softlabels.sbatch`, `export_ta_softlabels.sbatch`,
  `export_softlabels_swa_tta.sbatch`.
- **Quantization / INT8 export**: `train_qat_top2.sh`, `export_streaming_int8*.sh`.
- **Evaluation**: `eval_int8_full_val*.sh`, `eval_students.sbatch`,
  `swa_tta_eval.sbatch`, `eval_topk_c1j.sbatch`, `analyze_*`.
- **Early exit**: `early_exit_top2*.sh`, `early_exit_c1f_rerun.sh`.
- **ST Edge AI codegen**: `stedgeai_generate_sm_int8.sh`, `stedgeai_generate_top2.sh`
  (see [STEDGEAI_DEPLOYMENT.md](STEDGEAI_DEPLOYMENT.md)).

## Multi-node policy

Multiple nodes are used to run **more independent jobs in parallel** (SLURM arrays),
**not** for cross-node DDP. Cross-node DDP on the small models is dominated by
InfiniBand latency and is avoided. Intra-node DDP (NVLink, ≤4 GPU) is used only for
the Super Teacher and TA.

## I/O notes

The 9 Path-C readers share one `data/jester/jester_full.h5` (opened
`libver='latest', swmr=True`) and one `soft_labels_ta_train.npy` (small, fully page-
cached). Lustre file locking is disabled in `data_loader_tf.py` so concurrent reads
scale across the array.

## Logs

Per-job stdout/stderr land in `logs/<name>_<jobid>_<task>.out|.err`. These logs are
the source for the training-history summaries — see
[TRAINING_HISTORY_SUMMARY.md](TRAINING_HISTORY_SUMMARY.md) and `../results/README.md`.
