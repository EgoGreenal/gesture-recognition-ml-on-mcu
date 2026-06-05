# STM32U5 Gesture Recognition

This directory collects the STM32U5-specific deployment material for the gesture-recognition project.

Current contents:

- `mcu/firmware/gesture_c1j_U5_board/`
  - **Full STM32CubeIDE project (firmware `C1j-ai-v2`)** that runs the deployed C1j
    streaming INT8 model on the board (`ai_streaming_step`, single frame + TSM cache),
    with X-CUBE-AI generated network, host driver/benchmark scripts, and on-device
    `bench_*.json` results. This is the working on-device firmware.
- `firmware/main.c`
  - Earlier standalone smoke-test entry file (`C1j-smoke-v0`, board bring-up only,
    no inference). Superseded by the CubeIDE project above; kept for reference.
- `deploy/main_template.c`
  - Deployment-oriented `main.c` template for STM32CubeIDE / X-CUBE-AI integration.
- `deploy/host_inference_reference.c`
  - Host-side C reference for the intended streaming inference loop.
- `deploy/host_send_jester.py`
  - UART host tool for feeding Jester validation clips to the board.
- `docs/INSTALL.md`
  - HPC environment, venvs, toolchain, and dataset setup.
- `docs/TRAINING.md`
  - Three-stage TAKD pipeline (Super Teacher → TA → Path-C student) + quantization.
- `docs/EARLY_EXIT.md`
  - Early-exit strategies (S1/S3), deployed operating point, on-device benchmark.
- `docs/SERVER_TRAINING.md`
  - SLURM / Leonardo job inventory and conventions.
- `docs/MODEL_ARTIFACTS.md`
  - What is committed under `models/selected_deployable/` and how the pieces relate.
- `docs/STEDGEAI_DEPLOYMENT.md`
  - ST Edge AI Core INT8 C-code generation flow (U5 analog of MAX78000 AI8X_DEPLOYMENT).
- `docs/MCU_FIRMWARE.md`
  - On-device firmware (`C1j-ai-v2`): streaming inference, UART protocol, measured latency.
- `docs/HOST_DEMO.md`
  - Host-side UART harness (`host_benchmark.py` / `host_frame_test.py`); no live camera.
- `docs/TRAINING_HISTORY_SUMMARY.md`
  - Narrative of the V-line and C1-line model history (text form of the result CSVs).
- `docs/WORKFLOW.md`
  - Compact end-to-end summary of the STM32U5 training-to-deployment flow derived from the project plan.
- `docs/tsm_probe_results.md`
  - Probe report used to justify the streaming deployment path.
- `docs/day13_checklist.md`
  - STM32U5 deployment and validation checklist.
- `docs/final_report_skeleton.md`
  - Project final-report skeleton.
- `models/selected_deployable/`
  - Final chosen deployable TFLite models and generated ST Edge AI network sources.
- `results/`
  - Path selection, INT8 evaluation, early-exit, ST Edge AI summaries, training-history
    summaries, and `bench_summary.csv` (on-device C1j benchmark).
- `workspace/st_ai_output/`
  - Reserved folder for generated ST Edge AI outputs.
- `workspace/st_ai_ws/`
  - Reserved folder for STM32CubeIDE / X-CUBE-AI workspace files.

Notes:

- This part of the repository is deployment-focused and includes only the selected model artifacts and lightweight result summaries.
- Full training checkpoints, datasets, CubeIDE build artifacts, and large generated workspaces are intentionally not committed here.
- The STM32U5 track now covers the full path through **on-device streaming inference**
  (firmware `C1j-ai-v2` with measured ~141 ms/frame) and a host+board early-exit
  benchmark. The MAX78000 implementation in `../MAX78000/` still has the richer
  host-side demo (live camera) and a more complete doc set.
