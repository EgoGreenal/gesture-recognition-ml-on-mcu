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
  - Per-variant architecture analysis (params, MACs, RAM/ROM, INT8 size, estimated U5 latency).

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
