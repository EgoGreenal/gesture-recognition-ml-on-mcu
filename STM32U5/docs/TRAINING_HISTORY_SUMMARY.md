# TRAINING_HISTORY_SUMMARY — STM32U5 Track

Compact narrative of the U5 model history. Numbers are the text form of the machine
summaries in `../results/` (`model_training_summary.csv`,
`model_phase_history_summary.csv`, `confusion_summary.csv`, `bench_summary.csv`), all
derived from existing logs/artifacts — **no retraining**. See `../results/README.md`.

## Two lines of students

### Line V (V0–V6) — from-scratch tiny CNN+TSM — abandoned

A spatial/temporal model learned from scratch on 64×64 grayscale. All seven variants
topped out **below 40 %** FP32 val (best V4 = 38.45 %, most 15–25 %), 30 epochs each.
Conclusion: a tiny CNN lacks the spatial/temporal prior to learn 27-class video from
scratch. Abandoned in favour of distilling a truncated TSM-MobileNetV2.

### Line C1 — TSM-MBV2 truncated, KD from the TA — deployed

| Variant | Params | MMAC/frame | FP32 best | INT8 (no-exit, full val) | Role |
|---|---|---|---|---|---|
| C1f | 1.38 M | 17.46 | 87.35 % | **87.02 %** | backup deploy (best acc) |
| C1j | 0.72 M | 8.13 | 86.65 % | **86.49 %** | **primary deploy** (best size) |
| C1n | 0.70 M | 19.5 | 86.71 % | — | Path-C survivor |
| C1o | 1.55 M | 18.6 | 87.17 % | — | Path-C survivor |

Path-C selected survivors under the 600 KiB SRAM / 150 ms-per-frame budgets were
**C1f, C1j, C1n, C1o**; C1f and C1j were carried to INT8. C1j was chosen as the
primary deploy model for its smaller footprint at near-equal accuracy.

Training: KD from the TA's per-frame soft labels, Adam lr 1e-3, batch 64, alpha 0.3,
T 4.0. C1j: 60 epochs, best @ epoch 59. C1f: 30 epochs, best @ epoch 29.

## Quantization → deployment

- INT8 PTQ (clip, full 14,787-clip val, no exit): **C1f 87.02 %, C1j 86.49 %**.
- Per-class quality (`confusion_summary.csv`): C1f top-1 87.02 % / mean-class 86.26 %;
  C1j 86.49 % / 85.73 %.
- On-device (C1j streaming INT8, S1 early-exit thr 0.85 / mf 5, **n=128 subset**):
  **84.4 %**, mean exit frame 5.23, obs 0.778, **~141 ms/frame** measured.
  > The 84.4 % (n=128) and 86.49 % (full set, no exit) are **not directly comparable**
  > — see [EARLY_EXIT.md](EARLY_EXIT.md).

## Phases recorded per model (`model_phase_history_summary.csv`)

Each deployed model has `fp32_kd_train` (epochs, best epoch, last train loss, wall
time parsed from its SLURM log) and `int8_ptq` (INT8 accuracy) rows. Example C1j:
`fp32_kd_train` 60 ep, best @59, 86.65 %, ~10,744 s; `int8_ptq` 86.49 %.

## Teacher chain (context)

Super Teacher (Video Swin-T, video-level soft labels) → TA (TSM-MBV2, uni-directional,
per-frame soft labels) → C1 students. Details in [TRAINING.md](TRAINING.md).
