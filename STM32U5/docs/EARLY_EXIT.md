# EARLY_EXIT — STM32U5 Track

Early exit lets the recogniser emit a class decision **before** all 8 frames are
observed, trading a small accuracy drop for lower latency. On U5 this is the core
project goal (real-time gesture → light control).

## Current status

Early exit is currently implemented and evaluated **host-side** on the INT8 clip
TFLite models, to pick the operating point. The on-device decision loop (firmware
v1+, `ai_run(frame_t, cache) → logits_t` + threshold check) is **not yet built** —
see [tsm_probe_results.md](tsm_probe_results.md) for the streaming-interface probe
that gates it.

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

## Selected operating points

From `../results/early_exit/early_exit_winners.json` (budget: ≤ 1.0 pp drop vs the
no-exit INT8 baseline):

| Model | Baseline acc | Strategy | threshold | min_exit_frame | Exit acc | Mean obs. ratio | Acc drop |
|---|---|---|---|---|---|---|---|
| **C1f** | 87.02 % | S3 (entropy) | 0.8 | 5 | 86.16 % | 0.757 | 0.87 pp |
| **C1j** | 86.49 % | S3 (entropy) | 2.0 | 5 | 85.69 % | 0.750 | 0.80 pp |

Both winners land on **S3 with `min_exit_frame=5`**: on average they decide after
~6 of 8 frames (obs. ratio ~0.75) while staying within 1 pp of the full-clip
accuracy. The `latency_ms` field in the winners file is a host-side estimate; true
per-frame latency must be confirmed on the board once firmware v1+ exists.

Per-strategy full sweeps: `../results/early_exit/early_exit_C1{f,j}_ptq_v2.json`
(73 operating points each), comparison table `early_exit_comparison.csv`.
