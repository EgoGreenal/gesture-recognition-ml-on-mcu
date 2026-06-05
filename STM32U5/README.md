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
- `docs/WORKFLOW.md`
  - Compact end-to-end summary of the STM32U5 training-to-deployment flow derived from the project plan.
- `docs/tsm_probe_results.md`
  - Probe report used to justify the streaming deployment path.
- `models/selected_deployable/`
  - Final chosen deployable TFLite models and generated ST Edge AI network sources.
- `results/`
  - Path selection, INT8 evaluation, early-exit, and ST Edge AI summary artifacts.
- `workspace/st_ai_output/`
  - Reserved folder for generated ST Edge AI outputs.
- `workspace/st_ai_ws/`
  - Reserved folder for STM32CubeIDE / X-CUBE-AI workspace files.

Notes:

- This part of the repository is deployment-focused and includes only the selected model artifacts and lightweight result summaries.
- Full training checkpoints, datasets, CubeIDE build artifacts, and large generated workspaces are intentionally not committed here.
- The MAX78000 implementation in `../MAX78000/` remains the more complete end-to-end reference for training and demo flow.
