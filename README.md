# Gesture Recognition on MCUs

This repository contains a final course project for real-time hand gesture recognition on resource-constrained microcontrollers. The current complete implementation targets the Analog Devices MAX78000, and a placeholder is provided for an STM32U5 implementation that will be added by other contributors.

The project studies the full embedded ML pipeline:

- Training compact video gesture-recognition models on the 20BN-Jester dataset.
- Converting multi-frame video input into MCU-friendly 64x64 temporal representations.
- Running quantization-aware training for deployable int8 models.
- Synthesizing a MAX78000 CNN accelerator firmware project.
- Streaming live camera frames and predictions to a host-side realtime demo.
- Evaluating accuracy, model size, MMACs, inference time, and early-exit tradeoffs.

## Course Project Group Members
Yizhou Xu, Junkai Zhang

ETH Zürich

## Repository Layout

```text
gesture-recognition-ml-on-mcu/
  README.md
  LICENSE
  .gitignore
  MAX78000/
    README.md
    demo/
    docs/
    mcu/
    models/
    results/
    training/
  STM32U5/
    README.md
```

## MAX78000 Implementation

The MAX78000 part is the complete implementation in this repository:

- `MAX78000/training/`
  - Training, distillation, QAT, ai8x custom models/datasets/policies, server scripts, and experiment code.
- `MAX78000/mcu/firmware/stable/`
  - Standalone MSDK firmware project with generated CNN accelerator sources and deployed weights.
- `MAX78000/mcu/firmware/early_stop/`
  - Experimental early-exit firmware variant.
- `MAX78000/demo/`
  - Host-side realtime serial viewer for the camera demo.
- `MAX78000/models/`
  - Selected deployable model metadata and source model definition.
- `MAX78000/results/`
  - Lightweight training-history summaries, evaluation summaries, and structured report data.
- `MAX78000/docs/`
  - Installation, training, deployment, MCU firmware, host demo, early-exit, and model-history documentation.

Start from:

```text
MAX78000/README.md
```

## STM32U5 Implementation

The `STM32U5/` directory is intentionally a scaffold. Other contributors will add the STM32U5-specific training, quantization, deployment, and firmware material there.

## Dataset

The experiments use the 20BN-Jester v1 gesture dataset. The raw dataset is not included in this repository because of size and licensing constraints. Follow the platform-specific documentation under `MAX78000/docs/INSTALL.md` to prepare the dataset locally or on a training server.

## Large Files and External Dependencies

This public repository intentionally excludes:

- Raw datasets and cached frame tensors.
- PyTorch checkpoints and `.npy` sample tensors.
- Conda environments and Python virtual environments.
- External SDK/framework checkouts such as MSDK, ai8x-training, and ai8x-synthesis.
- Firmware build outputs such as `.elf`, `.bin`, `.hex`, `.map`, and object files.

The MAX78000 firmware remains buildable as a standalone MSDK project when an external MSDK installation is supplied through `MAXIM_PATH`.

## Reproducing the MAX78000 Demo

1. Read `MAX78000/docs/INSTALL.md`.
2. Prepare Python dependencies for the host demo.
3. Install the MAX78000 MSDK externally.
4. Build `MAX78000/mcu/firmware/stable`.
5. Flash the board with the generated firmware binary.
6. Run:

```powershell
cd MAX78000\demo
python realtime_camera_tk_demo.py --port COM5 --baud 2000000 --scale 6 --timeout 6 --threshold 90
```

Use the correct serial port for your system.

## Results Summary

The best deployable MAX78000 model in this repository is a 12-frame signed-motion residual CNN distilled from a larger software teacher and fine-tuned with QAT.

Key summary:

- Dataset: 20BN-Jester v1 validation split.
- Classes: 27 gestures.
- Input: 12 signed-motion grayscale frames, 64x64.
- Deployable topology: `jestermotion12res`.
- Parameters: 393,088.
- MMAC: 54.431968.
- QAT validation accuracy: 76.682% Top-1, 95.942% Top-5.
- MAX78000 CNN timing reference: about 4.701 ms for the deployable topology.

See `MAX78000/docs/TRAINING_HISTORY_SUMMARY.md` for the full model comparison table.
