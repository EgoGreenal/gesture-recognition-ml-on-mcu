# Training Pipeline

The training code targets compact video gesture-recognition models that can later be quantized and synthesized for MAX78000.

## Dataset representation

The source dataset is 20BN-Jester v1 with 27 gesture classes. The project uses the validation split for measured accuracy because the public test split does not include labels.

The main deployed representation is:

- 12 frames per sample.
- 64x64 grayscale resolution.
- Signed-motion input channels.
- 27 output logits.

The camera firmware keeps a sliding frame history and constructs the same 12-frame signed-motion tensor before loading data into the MAX78000 CNN SRAM.

## Core scripts

- `training/core/train_jester_cached.py`
  - Builds cached 64x64 frame tensors and trains compact CNN variants.
- `training/core/train_jester_distill.py`
  - Trains a compact student using a larger software teacher.
- `training/core/train_jester_multiframe.py`
  - Earlier multi-frame training entry point.
- `training/core/eval_jester_early_exit.py`
  - Evaluates early-exit policies on validation videos.
- `training/core/parse_jester_confusions.py`
  - Parses ai8x evaluation logs and confusion matrices.
- `training/core/remote_count_jester.py`
  - Utility for checking dataset and annotation counts on the remote server.

## Model families

The experiments include:

- 8-frame baseline.
- 12-frame raw grayscale models.
- 8-frame and 12-frame motion-channel models.
- Signed-motion models.
- Residual `jestermotion12res` variants.
- Wider or extra-convolution variants.
- Teacher-student distillation variants.
- Exploratory 3D CNN models for software-side comparison.

The selected deployable student is documented in `models/best_deployable/`.

## Distillation

The best deployable model was trained as a compact signed-motion residual student distilled from a larger teacher. Distillation improves the student without changing the MAX78000-compatible runtime topology.

## QAT

After floating-point or distilled training, the model is fine-tuned with ai8x QAT. The QAT policy for the selected topology is:

```text
models/best_deployable/model_definition/qat_policy_jestermotion12res.yaml
```

The final deployment uses signed 8-bit quantized weights and activations through the MAX78000 CNN accelerator.

## Outputs

Training and QAT normally produce:

- PyTorch checkpoints.
- Training logs.
- Confusion matrices.
- Sample `.npy` tensors for ai8x synthesis.
- Generated C sources from ai8xize.

Large checkpoints and generated run directories are excluded from this final source package. The firmware already contains generated `cnn.c`, `cnn.h`, and `weights.h` for the deployed model.

## Preserved summaries

The final project includes a compact all-model summary:

```text
docs/TRAINING_HISTORY_SUMMARY.md
results/model_training_summary.csv
results/model_phase_history_summary.csv
results/jester_training_report_data_20260530.json
```

The Markdown and CSV files summarize the 22 recorded model experiments and their 42 training phases. The JSON file keeps the fuller per-epoch training history used to generate the report.
