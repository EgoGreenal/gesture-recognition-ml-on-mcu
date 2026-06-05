# Results

This directory contains lightweight result summaries, not full training runs.

- `jester_training_report_data_20260530.json`
  - Full structured data used by the final report, including per-epoch histories.
- `model_training_summary.csv`
  - One row per recorded model, including best accuracy, params, MMAC, deployability, and source/config paths.
- `model_phase_history_summary.csv`
  - One row per training phase, including epoch count, best epoch, final validation metrics, and duration.
- `confusion_summary.csv`
  - Compact confusion/evaluation summary exported from the validation runs.

Large checkpoints, raw logs, cached frames, and generated run directories are intentionally excluded.
