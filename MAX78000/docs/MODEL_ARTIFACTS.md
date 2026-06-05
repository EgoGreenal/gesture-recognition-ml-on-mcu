# Model Artifacts

The selected deployable model metadata is stored under:

```text
models/best_deployable/
```

## Selected model

- Model ID: `ai8x_jestermotion12res_signed_huge_teacher_distill_20260530_223904`
- Runtime topology: `jestermotion12res`
- Input: 12 signed-motion grayscale frames, 64x64
- Output classes: 27
- Parameters: 393,088
- MMAC: 54.431968
- Distilled/pre-QAT validation: 77.757% Top-1, 95.476% Top-5
- QAT validation: 76.682% Top-1, 95.942% Top-5
- MAX78000 CNN timing reference for this topology: about 4.701 ms

## Included

- `metadata.json`
- Training/cache config JSON files.
- Label mapping.
- ai8x model definition.
- QAT policy.
- Generated firmware sources in `mcu/firmware/stable/`.
- Generated deployed weights in `mcu/firmware/stable/weights.h`.

## Excluded

Large PyTorch checkpoints are intentionally excluded from this source package:

- Pre-QAT `.pth` or `.pth.tar` files.
- QAT `.pth` or `.pth.tar` files.
- Generated `.npy` sample tensors.
- Full training run directories.

If checkpoints are needed, use the separate model package created for the best deployable model, or regenerate them using the server training scripts.
