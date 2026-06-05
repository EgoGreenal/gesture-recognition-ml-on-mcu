# Best Deployable MAX78000 Jester Gesture Model

Package created: 2026-06-01

## Selected model

- Model ID: `ai8x_jestermotion12res_signed_huge_teacher_distill_20260530_223904`
- Architecture: `jestermotion12res`
- Input: 12 signed-motion grayscale frames, 64x64
- Classes: 27 Jester gestures
- Parameters: 393,088
- MMAC: 54.431968
- Reason selected: best currently deployable MAX78000-compatible model from the experiment set. The larger `resxconv` variant reached slightly higher QAT validation accuracy, but was not treated as the stable deployable target.

## Validation results

- Distilled/pre-QAT best: Top-1 77.757%, Top-5 95.476%, epoch 8
- QAT best: Top-1 76.682%, Top-5 95.942%, epoch 15
- Board/topology timing reference: CNN accelerator inference around 4.701 ms for the `jestermotion12res` deployable topology.

## Contents

- `weights/pre_quant_student_distilled_best.pth.tar`
  - Distilled student checkpoint before QAT.
- `weights/post_quant_qat_best.pth.tar`
  - Best QAT checkpoint, intended as the post-quantization deployable weight artifact.
- `weights/post_quant_qat_checkpoint.pth.tar`
  - Last QAT checkpoint copied for reproducibility.
- `model_definition/jestermotion12res.py`
  - AI8X training model definition.
- `model_definition/qat_policy_jestermotion12res.yaml`
  - QAT policy used for this model family.
- `distill_artifacts/config.json`
  - Distillation/training configuration.
- `distill_artifacts/cache_config.json`
  - Dataset/cache preprocessing configuration.
- `distill_artifacts/labels.json`
  - Class label mapping.
- `logs/train.log`
  - Distillation/float training log.
- `logs/qat.log`
  - QAT training log.

## Original server locations

- Run directory: `/path/to/20bn-jester/runs/ai8x_jestermotion12res_signed_huge_teacher_distill_20260530_223904`
- Model definition: `/path/to/20bn-jester/ai8x-training/models/jestermotion12res.py`
- QAT policy: `/path/to/20bn-jester/ai8x-training/policies/qat_policy_jestermotion12res.yaml`

## Notes

This package intentionally contains both the pre-QAT distilled checkpoint and the post-QAT checkpoint. For a fresh MAX78000 deployment, use the QAT best checkpoint with the matching `jestermotion12res` model definition and the existing ai8x synthesis flow.
