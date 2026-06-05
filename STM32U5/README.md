# STM32U5 Gesture Recognition

This directory collects the STM32U5-specific deployment material for the gesture-recognition project.

Current contents:

- `firmware/main.c`
  - STM32U5 firmware entry file prepared for board-side integration and smoke-test bring-up.
- `deploy/main_template.c`
  - Deployment-oriented `main.c` template for STM32CubeIDE / X-CUBE-AI integration.
- `deploy/host_inference_reference.c`
  - Host-side C reference for the intended streaming inference loop.
- `deploy/host_send_jester.py`
  - UART host tool for feeding Jester validation clips to the board.
- `docs/day13_checklist.md`
  - STM32U5 deployment and validation checklist.
- `workspace/st_ai_output/`
  - Reserved folder for generated ST Edge AI outputs.
- `workspace/st_ai_ws/`
  - Reserved folder for STM32CubeIDE / X-CUBE-AI workspace files.

Notes:

- This part of the repository is currently deployment-focused rather than a full training pipeline.
- Generated binaries, CubeIDE build artifacts, datasets, and large model files are intentionally not committed here.
- The MAX78000 implementation in `../MAX78000/` remains the more complete end-to-end reference for training and demo flow.
