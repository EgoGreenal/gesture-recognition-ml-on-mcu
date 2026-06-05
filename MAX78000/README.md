# MAX78000 Real-Time Gesture Recognition

This folder is the standalone source package for the MAX78000 gesture-recognition project. It contains the project-owned training code, server launch scripts, MAX78000 firmware sources, host-side realtime demo, model definitions, and documentation.

The original `ex8` tree is intentionally not included. Large external dependencies such as MSDK, ai8x-training, ai8x-synthesis, Conda environments, cached datasets, compiled firmware outputs, and PyTorch checkpoints are also excluded. Install those dependencies separately by following `docs/INSTALL.md`.

## Project status

- Target board: MAX78000 with camera.
- Task: 27-class 20BN-Jester gesture recognition.
- Runtime model family: 12-frame signed-motion CNN for 64x64 grayscale input.
- Deployed inference: MAX78000 CNN accelerator, signed 8-bit quantized weights and activations.
- Stable firmware: `mcu/firmware/stable`.
- Experimental early-exit firmware: `mcu/firmware/early_stop`.
- Host GUI demo: `demo/realtime_camera_tk_demo.py`.

## Directory layout

```text
MAX78000/
  README.md
  docs/                         English documentation.
  training/
    core/                       Dataset caching, training, distillation, evaluation scripts.
    server_scripts/             Remote Linux/tmux scripts used on the training server.
    ai8x_custom_files/          Project-owned ai8x dataset/model/policy files.
    experiments/                Early-exit, 3D CNN, and report-generation experiments.
  models/
    best_deployable/            Metadata and source definition for the selected deployable model.
    notes/                      Architecture notes.
  results/                      Compact model summaries and training-history data.
  mcu/
    firmware/stable/            Standalone MSDK project with generated CNN sources and weights.
    firmware/early_stop/        Experimental early-exit firmware variant.
  demo/                         Standalone host serial viewer.
  tools/                        Reserved for lightweight helper scripts.
```

## Quick start

Install dependencies first:

```powershell
cd FinalProject\MAX78000
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install pyserial numpy pillow matplotlib
```

Run the host demo after flashing the board firmware:

```powershell
cd FinalProject\MAX78000\demo
python realtime_camera_tk_demo.py --port COM5 --baud 2000000 --scale 6 --timeout 6 --threshold 90
```

Build the stable firmware with an external MSDK installation:

```powershell
cd FinalProject\MAX78000\mcu\firmware\stable
make MAXIM_PATH=C:\path\to\msdk
```

On Linux, replace `C:\path\to\msdk` with the absolute path to the MSDK root.

## Main documents

- `docs/INSTALL.md`: host, MCU, server, and ai8x dependency setup.
- `docs/TRAINING.md`: dataset preprocessing, model training, distillation, and QAT.
- `docs/TRAINING_HISTORY_SUMMARY.md`: all-model summary table, phase histories, and source/YAML mapping.
- `docs/SERVER_TRAINING.md`: how the remote server scripts are organized and used.
- `docs/AI8X_DEPLOYMENT.md`: ai8x model integration, quantization, and C synthesis flow.
- `docs/MCU_FIRMWARE.md`: MAX78000 firmware structure and build/run notes.
- `docs/HOST_DEMO.md`: realtime serial demo and UI behavior.
- `docs/EARLY_EXIT.md`: early-exit policy, savings, and limitations.
- `docs/MODEL_ARTIFACTS.md`: selected model metadata and excluded checkpoint files.

## What is not included

- No original `ex8` tree.
- No MSDK checkout.
- No ai8x-training or ai8x-synthesis checkout.
- No Conda or Python virtual environments.
- No 20BN-Jester raw dataset, frame cache, or dataset archive.
- No compiled firmware binaries under `build/`.
- No `.pth`, `.pth.tar`, or other large PyTorch checkpoint files.

The stable firmware still builds independently as long as MSDK is installed externally and passed through `MAXIM_PATH`.
