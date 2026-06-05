# Host Demo

Run the recommended realtime GUI:

```powershell
python realtime_camera_tk_demo.py --port COM5 --baud 2000000 --scale 6 --timeout 6 --threshold 90
```

The demo expects the MAX78000 firmware to stream `New result` and `New image` packets over UART. It has no dependency on the original exercise directory.

See `../docs/HOST_DEMO.md`.
