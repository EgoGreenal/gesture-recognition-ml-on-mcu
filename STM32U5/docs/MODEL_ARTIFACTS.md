# MODEL_ARTIFACTS — STM32U5 Track

What is committed under `../models/selected_deployable/`, and how the pieces relate.
Source of truth: `../models/selected_deployable/metadata.json`.

## Deployed model

- **Primary variant: C1j** — TSM-MobileNetV2 truncated (family C1), 8 frames × 64×64
  grayscale, sequence input, per-frame KD. 721,979 params.
- **Backup variant: C1f** — same family, higher FP32/INT8 accuracy but larger Flash.

Training config (from `metadata.json`): 60 epochs, batch 64, Adam lr 1e-3, wd 1e-4,
KD alpha 0.3 / temperature 4.0, frame_weights `[0.1, 0.2, 0.3, 0.4]`, job `41478885_6`.

## Committed files

| File | Form | Purpose |
|---|---|---|
| `C1j_ptq_streaming_sm_int8.tflite` | streaming INT8 | **primary deploy** — single frame + 10 TSM caches I/O |
| `C1j_ptq_clip_int8.tflite` | clip INT8 | host-side eval / reporting (full 8-frame input) |
| `C1f_ptq_streaming_sm_int8.tflite` | streaming INT8 | backup deploy |
| `stedgeai_generate_int8/` | C package | ST Edge AI generated C for C1j (`student_c1j_ptq_int8.*`) |

The streaming vs clip distinction: the **streaming** export is what runs on the board
(true real-time / early-exit capable); the **clip** export takes all 8 frames at once
and is used only for host evaluation. Both quantize the same trained weights.

## ST Edge AI footprint (recorded in `metadata.json`)

| Metric | Value |
|---|---|
| MACC | 8,132,051 / frame |
| RAM (activations) | 267,328 B (≈261 KiB) |
| Weights | 2,850,828 B |
| Estimated latency | 32.11 ms/frame |

> The 32.11 ms/frame is the **offline ST Edge AI estimate**. The board measured
> **~141 ms/frame** — see [EARLY_EXIT.md](EARLY_EXIT.md) and `../results/bench_summary.csv`.

## Accuracy lineage

- C1j FP32 best val (obs 1.0): **86.65 %**
- C1j clip INT8 full-val (no exit): **86.49 %**
- Early-exit winner under ≤1 pp drop (offline analysis): S3, thr 2.0, mf 5 → 85.69 %
- **Deployed on-device operating point: S1, thr 0.85, mf 5** (see EARLY_EXIT.md). The
  `early_exit_best_under_1pp_drop` block in `metadata.json` records the S3 offline
  winner; the shipped firmware uses the simpler S1 @ 0.85.

## Not committed

Raw training checkpoints (`*.weights.h5`), datasets, soft-label `.npy`, and X-CUBE-AI
build outputs are excluded (`.gitignore`). The on-board integration of the generated C
lives in `../mcu/firmware/gesture_c1j_U5_board/X-CUBE-AI/` — see
[MCU_FIRMWARE.md](MCU_FIRMWARE.md).
