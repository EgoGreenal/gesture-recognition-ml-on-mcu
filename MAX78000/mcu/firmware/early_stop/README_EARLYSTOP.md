# Jester Sliding Early-Stop Firmware Variant

This directory is a separate firmware variant copied into the final MAX78000 project.
It is intentionally not flashed by default.

## Policy

- The camera history is still updated every frame at the paced 12 fps loop.
- Every non-held sliding-window frame first runs the CNN with prefix 8.
- If the prefix-8 prediction passes the class-specific confidence threshold, the
  firmware holds that prediction and skips CNN calls for the next 4 frames.
- Otherwise, the same frame runs the CNN again with prefix 10.
- If the prefix-10 prediction passes the threshold, the firmware holds that
  prediction and skips CNN calls for the next 2 frames.
- Otherwise, the same frame runs the full 12-frame model.

This keeps the sliding-window camera stream.  Only the held frames skip CNN
invocations; non-held frames may use one, two, or three CNN calls depending on
whether prefix 8 or prefix 10 is accepted.

## Build

From this directory on a Windows setup with the Maxim/Analog Devices MSDK installed:

```powershell
make MAXIM_PATH=C:\path\to\msdk
make build/max78000.bin MAXIM_PATH=C:\path\to\msdk
```

The output binary is:

```text
build/max78000.bin
```

Do not copy it to the DAPLink drive unless you intentionally want to flash this
variant.
