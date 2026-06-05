# Host Realtime Demo

The host demo reads serial packets emitted by the MAX78000 firmware and displays the live camera image, prediction, confidence, timing, FPS, and gesture-control lamp state.

## Files

- `demo/realtime_camera_demo.py`
  - Packet parser and matplotlib/non-GUI viewer.
- `demo/realtime_camera_tk_demo.py`
  - Recommended Tkinter UI.
- `demo/run_qat_demo.bat`
  - Windows convenience launcher using the current directory and system Python.

The same Python files are also copied next to each firmware variant for convenience, but the standalone demo under `demo/` is the recommended entry point.

## Run

```powershell
cd FinalProject\MAX78000\demo
python realtime_camera_tk_demo.py --port COM5 --baud 2000000 --scale 6 --timeout 6 --threshold 90
```

Use the correct serial port for your board. On Linux the port is usually similar to `/dev/ttyACM0`:

```bash
python realtime_camera_tk_demo.py --port /dev/ttyACM0 --baud 2000000 --scale 6 --timeout 6 --threshold 90
```

## UI behavior

- `Prediction`, `Class`, and `Confidence` show the current model output.
- `Threshold` controls the minimum confidence used for accepting non-negative gesture commands.
- `Last Accepted` stores the latest accepted non-negative gesture.
- Four lamp boxes simulate direction-control output.
- `Thumb Down` clears all lamps.
- `Inference FPS`, `Display FPS`, and `CNN Time` show backend and frontend timing.
- Optional profile fields show capture, preprocessing, input loading, softmax, and frame time when the firmware emits them.

## Packet protocol

The firmware emits `New result` packets for prediction-only updates and `New image` packets for image plus result updates. The parser supports:

- `RGB888`
- `RGB565`
- `GRAY8`

## Smoke test

To parse a few packets without opening a GUI:

```powershell
python realtime_camera_demo.py --port COM5 --baud 2000000 --no-display --frames 3
```

Use `--save-dir` to save received frames for debugging.
