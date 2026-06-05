# Installation Guide

This project keeps external dependencies out of the source package. Install the required tools separately, then point the scripts or build commands to those installations.

## Host demo environment

The host demo only needs Python and a serial connection to a flashed MAX78000 board.

```powershell
cd FinalProject\MAX78000
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install pyserial numpy pillow matplotlib
```

On Linux or macOS:

```bash
cd FinalProject/MAX78000
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pyserial numpy pillow matplotlib
```

## MCU firmware build environment

Install the Maxim/Analog Devices MSDK outside this folder. The firmware projects expect the MSDK root through `MAXIM_PATH`.

Typical requirements:

- MSDK for MAX78000.
- Arm embedded GCC toolchain compatible with the MSDK build system.
- `make` or `mingw32-make` on Windows.
- Board USB drivers and DAPLink/OpenOCD support if flashing or debugging.

Build example:

```powershell
cd FinalProject\MAX78000\mcu\firmware\stable
make MAXIM_PATH=C:\path\to\msdk
```

The firmware folder does not include MSDK headers, drivers, board support packages, or tools. This is deliberate.

## ai8x training and synthesis tools

Install ai8x-training and ai8x-synthesis outside this folder:

```bash
git clone https://github.com/analogdevicesinc/ai8x-training.git
git clone https://github.com/analogdevicesinc/ai8x-synthesis.git
```

Then install the Python dependencies requested by those repositories in your own Conda or virtual environment. The custom dataset, model, and QAT policy files from this project are under:

```text
training/ai8x_custom_files/codex_uploads/
models/best_deployable/model_definition/
```

Copy those files into the matching `datasets/`, `models/`, and `policies/` directories of your ai8x-training checkout when reproducing the QAT runs.

## Remote server environment

The original remote experiments used:

```bash
ssh <user>@<server>
conda activate ai4eda
cd /path/to/20bn-jester
```

The shell scripts in `training/server_scripts/` preserve those original absolute paths. If you move the project to another server, edit the `PROJECT`, `DATASET_ROOT`, or `DATASET` variables at the top of the relevant scripts.

## Dataset

The 20BN-Jester dataset is not included. The scripts expect the extracted dataset directory to contain the frame/video folders and annotation CSVs used by the Kaggle-hosted 20BN-Jester v1 release.

The preprocessing code resizes frames to 64x64 and can generate raw-frame, motion-channel, signed-motion, and cached variants depending on the script and experiment.
