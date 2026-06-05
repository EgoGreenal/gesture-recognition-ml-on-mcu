# STM32U5 Results

This directory contains lightweight summary artifacts used to justify the STM32U5 deployment choices.

- `path_selection/`
  - Path-C summary and shortlisted variants under memory and latency constraints.
- `int8_eval/`
  - Host-side INT8 evaluation summaries for the selected C1 variants.
- `early_exit/`
  - Early-exit sweep summaries and aggregate comparison outputs.
- `stedgeai/`
  - ST Edge AI analysis summaries used to estimate STM32U5 deployment cost.
- `analyze_summary.json`
  - Per-variant architecture analysis (params, MACs, RAM/ROM, INT8 size, **estimated**
    U5 latency — see the measured-latency caveat below).
- `bench_summary.csv`
  - **On-device** benchmark of the deployed C1j streaming INT8 model (S1 @ 0.85,
    min_exit_frame 5), extracted from `../mcu/firmware/gesture_c1j_U5_board/bench_*.json`:
    accuracy, mean exit frame, observation ratio, measured ms/frame.
  - **Sample size:** these are **small on-device test subsets** (max 128 clips →
    84.4 %), *not* the full set. The **86.49 %** in `int8_eval/` and
    `model_training_summary.csv` is the **full 14,787-clip** no-exit evaluation. The
    two are not directly comparable (n=128 sampling variance + early-exit drop). See
    `../docs/EARLY_EXIT.md`.

> **Estimated vs measured latency.** The `latency`/`lat_per_frame_ms` fields in
> `analyze_summary.json`, `model_training_summary.csv`, and `path_selection/` are
> **offline X-CUBE-AI estimates** (~32–70 ms/frame). The real board measures
> **~141 ms/frame** for C1j (`bench_summary.csv`) — the estimate under-predicted by
> ~2–4× (effective Helium INT8 throughput ~57 MMAC/s vs the assumed ~300). Treat the
> estimated columns as lower bounds; `bench_summary.csv` is ground truth.

## Training-history summaries

Project-wide summaries mirroring the MAX78000 result schema. **Derived from existing
artifacts only — no retraining.** Built by parsing training logs (`logs/*.out`, linked
to each model via its checkpoint job id), the structured result JSONs
(`analyze_summary.json`, `path_c_summary.json`, `eval_int8_*.json`), and the saved
validation logits (`early_exit_*_logits.npz`).

- `confusion_summary.csv` — per-model top-1 and mean per-class accuracy on the 14,787-clip
  Jester val set, computed from saved INT8 logits for the deployed C1f / C1j models.
- `model_training_summary.csv` — one row per variant (V0-V6 from-scratch line + Path-C
  survivors C1f/C1j/C1n/C1o): params, MACs, FP32 best top-1, INT8 top-1, INT8 size,
  estimated U5 latency, training epochs/time, checkpoint path.
- `model_phase_history_summary.csv` — per-phase history (`fp32_kd_train`, `int8_ptq`)
  with best epoch, last train loss, and duration parsed from the source log.

The large raw logs, full run directories, and board-generated outputs are intentionally excluded.
