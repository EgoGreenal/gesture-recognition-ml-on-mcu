# MCU Firmware

The firmware is packaged as a standalone MSDK project. It does not depend on the original exercise directory.

## Variants

- `mcu/firmware/stable/`
  - Main realtime gesture-recognition firmware.
  - Uses the generated `jestermotion12res_signed_qat` CNN accelerator sources.
  - Runs a 12 fps inference loop and emits serial packets for the host demo.
- `mcu/firmware/early_stop/`
  - Experimental early-exit variant.
  - Preserved for energy-saving experiments.
  - Not the default firmware to flash.

## Build

Install MSDK externally, then build:

```powershell
cd FinalProject\MAX78000\mcu\firmware\stable
make MAXIM_PATH=C:\path\to\msdk
```

The expected output is:

```text
build/max78000.bin
```

On Linux:

```bash
cd FinalProject/MAX78000/mcu/firmware/stable
make MAXIM_PATH=/path/to/msdk
```

## Runtime behavior

The stable firmware:

- Captures camera frames.
- Converts RGB565 camera pixels to grayscale.
- Resizes/uses a 64x64 inference view.
- Maintains a 12-frame sliding history.
- Builds signed-motion CNN input tensors.
- Loads input data into the MAX78000 CNN SRAM.
- Starts the CNN accelerator.
- Reads 27 logits.
- Runs fixed-point softmax on the CPU.
- Sends prediction, timing metadata, and periodic image packets over UART.

The firmware targets:

- 12 fps backend inference cadence.
- 4 fps foreground image display cadence.
- 2,000,000 baud serial output by default.

## Timing

`cnn_time` is measured with the generated CNN timer hook around the accelerator invocation. Additional profiling fields in `main.c` measure capture, history update, input load, softmax, post-processing, and total frame time.

## Flashing

Flashing is not automated in this source package. Use the MSDK-supported method for your board, for example DAPLink copy or OpenOCD, after building `build/max78000.bin`.
