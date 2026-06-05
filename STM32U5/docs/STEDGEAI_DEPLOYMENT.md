# STEDGEAI_DEPLOYMENT — STM32U5 Track

U5 counterpart of the MAX78000 `AI8X_DEPLOYMENT.md`. Where MAX78000 uses the
ai8x-synthesis toolchain to map a network onto the MAX78000 CNN accelerator, the U5
track uses **ST Edge AI Core 4.0** (= STM32Cube.AI 12.0) to generate C code that runs
on the Cortex-M33 with CMSIS-NN / Helium.

## Pipeline

```
TF/Keras student  →  INT8 TFLite (PTQ)  →  ST Edge AI `analyze` / `generate`  →  C code
   (training)         streaming + clip       (estimate cost)   (network .c/.h)     (→ CubeIDE)
```

1. **Quantize** to INT8 TFLite (see [TRAINING.md](TRAINING.md) §5). Two exports:
   `*_streaming_sm_int8.tflite` (single frame + TSM cache I/O, for deployment) and
   `*_clip_int8.tflite` (8-frame, for host eval).
2. **Probe** the streaming multi-input/multi-output graph quantizes and is accepted by
   the toolchain — see [tsm_probe_results.md](tsm_probe_results.md). This gated the
   whole streaming deployment path.
3. **Generate** C code with `stedgeai generate`. Wrapper:
   `../training/server_scripts/sbatch/stedgeai_generate_sm_int8.sh` (SLURM array over
   C1f/C1j, on the **real** INT8 streaming TFLite — an early run accidentally picked up
   an FP32 fallback, hence the rerun note in the script).

## Toolchain setup

`scripts/env_setup.sh` appends `tools/stedgeai/4.0/Utilities/linux` to `PATH`. It is
**appended, not prepended**, because the install ships a bundled Python 3.9.13 that
would otherwise shadow the active venv. See [INSTALL.md](INSTALL.md).

## Outputs

- Generated C package: `../models/selected_deployable/stedgeai_generate_int8/`
  (`student_c1j_ptq_int8.c/.h`, `_data.c`, `_details.h`, `_c_info.json`,
  `_generate_report.txt`). This is dropped into the CubeIDE project under
  `../mcu/firmware/gesture_c1j_U5_board/X-CUBE-AI/App/`.
- Analyze summaries: `../results/stedgeai/stedgeai_C1{f,j}_ptq.json`.

## Cost (from the generate report / `metadata.json`)

| | C1j streaming INT8 |
|---|---|
| MACC | 8,132,051 / frame |
| RAM (activations) | 267,328 B (≈261 KiB) |
| Weights | 2,850,828 B |
| **Estimated latency** | 32.11 ms/frame |

> **Estimate vs silicon.** The 32.11 ms/frame estimate assumed ~300 MMAC/s; the board
> measured **~141 ms/frame** (effective ~57 MMAC/s). The offline estimate is a lower
> bound — see [EARLY_EXIT.md](EARLY_EXIT.md) and `../results/bench_summary.csv`. Make
> sure the cmsis-nn / Helium optimisation is enabled in the X-CUBE-AI settings, or it
> is even slower.

## Constraints met

Flash < 2 MB, RAM < 600 KiB, per-frame latency < 150 ms — all pass for C1j (Flash
≈0.99 MB, RAM ≈261 KiB, 141 ms/frame). See [MODEL_ARTIFACTS.md](MODEL_ARTIFACTS.md).
