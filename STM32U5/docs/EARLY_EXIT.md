# EARLY_EXIT — STM32U5 Track

Early exit lets the recogniser emit a class decision **before** all 8 frames are
observed, trading a small accuracy drop for lower latency. On U5 this is the core
project goal (real-time gesture → light control).

## Current status

Early exit runs in a real **board + host** loop:

- **Board** (firmware `C1j-ai-v2`, `../mcu/firmware/gesture_c1j_U5_board/`) runs the
  streaming model one frame at a time: `ai_streaming_step(frame_t, logits_t, cycles)`
  keeps the TSM cache across calls (the A.streaming path validated by the
  [tsm_probe](tsm_probe_results.md)). It emits per-frame logits + DWT cycle count.
- **Host** (`host_benchmark.py`) applies the stopping rule over the streamed
  per-frame logits and stops sending frames once the exit condition is met.

So the per-frame inference is measured on real silicon; the lightweight exit
decision is taken host-side. Folding the decision into the firmware itself is the
only remaining step (it needs ~10 lines: a softmax-max + threshold check).

## Strategies (`training/core/early_exit_eval.py`)

Per-frame logits are collected from the clip model, then a stopping rule is applied
frame by frame:

- **S1 — max-softmax**: exit at the first frame `t` where `max(softmax(logits[t])) > threshold`.
- **S3 — entropy**: exit at the first frame `t` where `entropy(softmax(logits[t])) < threshold`.

Both are floored by **`min_exit_frame`**: exits before that frame are forbidden
(early frames are unreliable — a uni-directional TSM has seen too little).

Sweep + aggregation:
- `early_exit_eval.py` — sweeps the threshold grid per strategy, records accuracy,
  mean observation ratio, latency, and the exit-frame histogram.
- `build_early_exit_summary.py` — picks winners under a max-accuracy-drop budget.
- `plot_early_exit.py` — accuracy-vs-observation curves
  (`../results/early_exit/early_exit_curve_v2.png`).

## Deployed operating point — S1, threshold 0.85, min_exit_frame 5

The **deployed** model is **C1j** (streaming INT8) and the **deployed early-exit rule
is S1 (max-softmax) with `threshold = 0.85`, `min_exit_frame = 5`**. This is the
operating point wired into `host_benchmark.py` and used for the on-device benchmark
([day13_checklist.md](day13_checklist.md), report TL;DR).

### On-device benchmark (`../mcu/firmware/gesture_c1j_U5_board/bench_*.json`)

Real board runs over **small Jester test subsets** (the board is fed clips one at a
time over UART, so these are sized for turnaround, not full coverage), S1 @ 0.85 / mf 5:

| Bench | clips | Accuracy | Mean exit frame | Mean obs. ratio |
|---|---|---|---|---|
| bench_10  | 10  | 80.0 % | 5.20 | 0.775 |
| bench_100 | 100 | 84.0 % | 5.27 | 0.784 |
| bench_128 | 128 | **84.4 %** | 5.23 | 0.778 |

> **Sample-size caveat — not directly comparable to the 86.49 % figure.** The
> **84.4 %** here is on the **largest on-device run of 128 clips**. The **86.49 %**
> quoted elsewhere is the **full 14,787-clip** evaluation (no early exit). The 128-clip
> on-device number carries large sampling variance (±~6 pp at n=128) and additionally
> reflects the early-exit drop and the streaming-vs-clip INT8 export, so the ~2 pp gap
> is expected and the two numbers must not be read as a regression. A full-set on-device
> run is left as future work (it needs the firmware-side exit decision to avoid streaming
> all 14,787 clips over UART).

**Measured latency: ~141 ms/frame** of pure compute (≈22.6 M Cortex-M33 cycles @
160 MHz, from DWT) — within the 150 ms/frame budget. With early exit at ~5.2 frames
the per-clip wall time is ~880 ms including the 4 KB/frame UART transfer.

> **Estimated vs measured.** The offline X-CUBE-AI per-frame estimate (≈32–70 ms/frame
> in `../results/model_training_summary.csv` / `path_selection/`) under-predicted real
> silicon by ~2–4×: effective Helium INT8 throughput was ~57 MMAC/s, not the assumed
> ~300 MMAC/s. Treat the estimated-latency columns as lower bounds.

## Offline strategy analysis (full 14,787-clip val set)

`training/core/early_exit_eval.py` sweeps both strategies on the host over the full
val set; `build_early_exit_summary.py` writes winners under a ≤ 1.0 pp drop budget
(`../results/early_exit/early_exit_winners.json`):

| Model | Baseline acc | Strategy | threshold | min_exit_frame | Exit acc | Mean obs. | Acc drop |
|---|---|---|---|---|---|---|---|
| C1f | 87.02 % | S3 (entropy) | 0.8 | 5 | 86.16 % | 0.757 | 0.87 pp |
| C1j | 86.49 % | S3 (entropy) | 2.0 | 5 | 85.69 % | 0.750 | 0.80 pp |

- **Strategies.** S1 = max-softmax (`max(softmax(logits[t])) > threshold`);
  S3 = entropy (`entropy(softmax(logits[t])) < threshold`). Both floored by
  `min_exit_frame` (early frames are unreliable for a uni-directional TSM).
- The winners file's `latency_ms` field is a host-side estimate — superseded by the
  on-device numbers above.
- **Why S1 on device, S3 in the winners file:** the winners file picks the
  lowest-drop point on the full val set (S3); the deployed firmware uses the simpler
  S1 @ 0.85 (one `max()` + compare, no per-frame entropy) at a near-identical obs.
  ratio (~0.78). The on-device 84.4 % (n=128) vs the offline S3 85.7 % (n=14,787)
  gap reflects the strategy choice, the much smaller on-device sample, and the
  streaming-vs-clip INT8 export.

Per-strategy full sweeps: `../results/early_exit/early_exit_C1{f,j}_ptq_v2.json`
(73 operating points each), comparison table `early_exit_comparison.csv`.
