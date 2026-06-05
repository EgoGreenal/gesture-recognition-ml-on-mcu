@echo off
setlocal
cd /d "%~dp0"
python realtime_camera_tk_demo.py --port COM5 --baud 2000000 --scale 6 --timeout 6 --threshold 90
pause
