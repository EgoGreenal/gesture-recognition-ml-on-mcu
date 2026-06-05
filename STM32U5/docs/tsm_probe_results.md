# Day 8 TSM Deployment-Path Probe Report

**Date**: 2026-05-12
**Goal**: Decide the Day 13 deployment path — Path A.streaming (single TFLite + cache-as-IO) vs Path B (multi-network fallback)
**Verdict**: ✅ **Path A.streaming — GO** (A.streaming is the primary path; all 7 Day 9-10 variants V0-V6 can proceed)

---

## Probe model design

Minimal viable streaming student (**deliberately tiny** — only to validate the topology
+ quantization path; not a student-variant candidate):

```
frame_in [1, 32, 32, 1]                                ch=1   spatial 32x32
   │
   │  Conv2D(8, 3, stride=2) → BN → ReLU                ch=8   spatial 16x16
   │
   │  TSMStreamingShift  ← cache_0_in [1, 16, 16, 1]    ch=8 (n_shift=1)
   │                    → cache_0_out [1, 16, 16, 1]
   │  Conv2D(16, 3, stride=2) → BN → ReLU               ch=16  spatial 8x8
   │
   │  TSMStreamingShift  ← cache_1_in [1, 8, 8, 2]      ch=16 (n_shift=2)
   │                    → cache_1_out [1, 8, 8, 2]
   │  Conv2D(32, 3, stride=2) → BN → ReLU               ch=32  spatial 4x4
   │
   │  GlobalAveragePool                                 (1, 32)
   │  Dense(27, no activation)                          (1, 27) logits
   ▼
[logits, cache_0_out, cache_1_out]
```

**Key design points**:
- Input channels are chosen so that `n_shift = C / n_div = C/8` is an integer at every
  TSM boundary (8/8=1, 16/8=2) — avoids the TFLite Slice quantization scale-misalignment
  corner case.
- Each `Conv → BN → ReLU` triple ends with ReLU, and the TSM cut point is after the ReLU
  — satisfies the Plan Day 8 "BN folding boundary rule".
- TSM shift is implemented with pure `tf.split + tf.concat`, no custom op needed.

See [../training/core/tsm_streaming_layer.py](../training/core/tsm_streaming_layer.py) +
[../training/core/tsm_probe.py](../training/core/tsm_probe.py).

---

## Check 1 — INT8 quantization fidelity

**Goal**: after FP32 Keras → INT8 TFLite (post-training quantization), the output
distribution should deviate by < KL 0.05.

**Procedure**:
1. Keras model → `tf.saved_model.save` → `tf.lite.TFLiteConverter.from_saved_model`
2. `representative_dataset` yields dict-keyed samples (multi-input calibrator input-order pitfall)
3. `target_spec.supported_ops = [TFLITE_BUILTINS_INT8]` + `inference_input_type = INT8` +
   `inference_output_type = INT8` → fully quantized I/O
4. 32 random [-1, 1] uniform samples, each run through FP32 Keras + the INT8 TFLite
   Interpreter, comparing the softmax KL of the logits

**Results**:

| Metric | Value | Note |
|---|---|---|
| samples | 32 | |
| KL(FP32 ‖ INT8) | **0.000001** | threshold < 0.05 ✓ |
| KL(INT8 ‖ FP32) | 0.000001 | symmetric check |
| argmax agreement | **1.0000** | all 32 samples predict identically |

**verdict: PASS** ✓

Note: the model is untrained (random init), so the output logits are near-uniform and
quantization has almost nothing to lose. **What this validates is the topology + the
quantization path itself**, not accuracy. A real trained student will see INT8 KL rise
but stay well below 0.05 (Day 11 QAT tightens it again).

---

## Check 2 — stedgeai analyze (X-CUBE-AI multi-I/O + op support)

**Goal**: can ST Edge AI Core 4.0 (= STM32CubeAI 12.0):
1. Accept a multi-input/multi-output INT8 TFLite (3 inputs / 3 outputs)?
2. Quantize the TSM STRIDED_SLICE + CONCATENATION faithfully?
3. Produce MAC / RAM / ROM estimates?

**Command**:
```bash
stedgeai analyze \
    --model streaming_probe_int8.tflite \
    --target stm32 \
    --name streaming_probe \
    --workspace logs/tsm_probe/stedgeai/workspace \
    --output logs/tsm_probe/stedgeai/output \
    --verbosity 2
```

**Results**:

| Metric | Value |
|---|---|
| stedgeai version | ST Edge AI Core v4.0.0-20500 (STM32CubeAI 12.0.0) |
| returncode | **0** ✓ |
| multi-I/O accepted | **True** ✓ |
| model type | tflite (fully INT8, `ss/sa per channel`) |
| model hash | `0x96cc70c7739a60adb4adc6f87ad69071` |
| params # | 6,752 items (**6.76 KiB**) |
| inputs total | 1.38 KBytes (frame 1024 + cache_0 256 + cache_1 128) |
| outputs total | 411 Bytes (logits 27 + cache_0_out 256 + cache_1_out 128) |
| MACs | **170,419** |
| weights (RO) | 7,028 B (**6.86 KiB**) — -74.0% vs the FP32 model |
| activations (RW) | 11,136 B (**10.88 KiB**) |
| RAM total | 11,136 B (10.88 KiB) |
| ROM (AI RT code) | not given (needs `arm-none-eabi-gcc` on PATH to size the runtime code; install the ARM toolchain before the Day 13 board flash) |
| tool time | 20.82 s |
| analyze progress | 162/171 PASS (remaining 9 skipped because GCC was not installed) |

**verdict: PASS** ✓

**stedgeai output excerpt** (the key topology is recognised):
```
m_id   layer (original)                  oshape                 param/size    macc
0      serving_default_frame_in0         [b:1,h:32,w:32,c:1]
       conv2d_0 (CONV_2D)                [b:1,h:16,w:16,c:8]    80/104         18,440
       nl_0_nl (CONV_2D)                 [b:1,h:16,w:16,c:8]                    2,048   <- ReLU as fused-conv
1      slice_1 (STRIDED_SLICE)           [b:1,h:16,w:16,c:1]                            <- TSM cache slice
2      slice_2 (STRIDED_SLICE)           [b:1,h:16,w:16,c:7]                            <- TSM static slice
3      concat_3 (CONCATENATION)          [b:1,h:16,w:16,c:8]                            <- TSM re-join
...
```

**Key findings**:
1. **Multi-I/O fully OK**: 3 inputs (frame + 2 cache) and 3 outputs (logits + 2 cache_out)
   are all recognised as normal tensors; no unsupported-op error triggered.
2. **TSM SLICE/CONCAT preserved cleanly**: each TSM boundary decomposes into
   `STRIDED_SLICE×2 + CONCATENATION×1`, which stedgeai recognises as standard ops and
   quantizes (per-channel ss/sa scale).
3. **Quantization scale auto-synced across SLICE/CONCAT nodes**: cache_0_in and cache_0_out
   are both `QLinear(0.008567031, -11, int8)` — identical scale + zero point, meaning the
   cache buffer can be **memcpy'd directly** in C with no rescale.
4. **Plan section 3 requires ≥ X-CUBE-AI 9.1 (STM32CubeAI 9.x)**; the installed STM32CubeAI
   12.0 is newer with more complete op support.

---

## Check 3 — on-board DWT measurement (U5)

**Status**: **skipped** (no U5 board on HPC Leonardo at the time)

**Replaced by the estimate formula** (Plan Section 5 corrected formula):
```
latency_ms = MACs / 300e6 × 1000 + N_ai_run_calls × 5ms
```
- Probe model MACs = 170,419 ≈ 0.17M
- 8 frames × 1 ai_run/frame = 8 ai_run calls
- Estimate: `0.17M / 300M × 1000 + 8 × 5 = 0.6 + 40 = ~40.6 ms` for an 8-frame clip

The probe model is deliberately tiny, so its latency does not represent the real V0-V6
numbers. But the MACs stedgeai reports are **per-frame**, not per-clip; run the same
analyze on each Day 9-10 variant to get real MACs, then plug into the formula for the
per-frame latency.

> **Update (Day 13):** the on-board measurement was later done on the deployed C1j
> firmware — **~141 ms/frame**, ~2-4× higher than the 300 MMAC/s estimate above. The
> estimate is a lower bound. See [EARLY_EXIT.md](EARLY_EXIT.md) and
> [../results/bench_summary.csv](../results/bench_summary.csv).

---

## Decision and next steps

### Deployment path locked: **Path A.streaming**

Day 13 deployment uses a single TFLite + C-side cache buffer management, one `ai_run()`
per frame, with only ~5 ms × 8 = 40 ms per-frame overhead. The multi-network fallback
(Path B) is **not** needed, and V0/V1 do **not** need to be shrunk for it.

### Day 9-10 variant pool opened to all 7

Per the Plan Section 5 table, under the A.streaming single-NN path:

| Variant | Estimated latency | Feasible |
|---|---|---|
| V0 Ultra-Tiny | 140 ms | ✓ |
| V1 Tiny | 97 ms | ✓ |
| V2 Small | 307 ms | ⚠ over 200ms |
| V3 Small-DW | 190 ms | ✓ borderline |
| V4 Deep | 373 ms | ✗ |
| V5 Wide | 973 ms | ✗ |
| V6 HiRes | 473 ms | ✗ |

Real training estimates may be ±30%. After Day 9-10 FP32 training, run `stedgeai analyze`
once per variant for real MACs / RAM, and **use the real numbers to pick the Day 11 QAT top-3**.

### Fallback work not needed
- ❌ No need to implement `block_exporter.py` (multi sub-network splitting — Path B only)
- ❌ No need to insert identity FakeQuant + custom `tfmot.QuantizeConfig` at block
  boundaries (QAT scale consistency — Path B only)
- ❌ No need to write `tsm_shift.c` (C-side memcpy shift — Path B only)
- ✅ Still needed: student model factory + frame-level KD loss + training scripts

---

## Probe tool outputs

- [../training/core/tsm_streaming_layer.py](../training/core/tsm_streaming_layer.py) —
  TSMStreamingShift Keras layer + `causal_tsm_clip` clip-form helper (clip-form for
  training, unrolled into streaming steps for deployment)
- [../training/core/tsm_probe.py](../training/core/tsm_probe.py) — full probe pipeline
  (build → INT8 export → Interpreter cross-check → stedgeai analyze → JSON report)
- `logs/tsm_probe/` — full output directory:
  - `streaming_probe_fp32.tflite` — 30.6 KB
  - `streaming_probe_int8.tflite` — 12.9 KB
  - `saved_model/` — TF SavedModel intermediate format
  - `stedgeai/workspace/` + `stedgeai/output/` — full stedgeai working dir
  - `stedgeai/analyze_log.txt` — full stedgeai run log
  - `tsm_probe_results.json` — machine-readable result summary
- `tools/stedgeai/4.0/` — full ST Edge AI Core 4.0 (= STM32CubeAI 12.0) install tree (not committed)
- [../training/server_scripts/env_setup.sh](../training/server_scripts/env_setup.sh) —
  adds `stedgeai` to PATH (append-not-prepend, to avoid the bundled python 3.9.13
  shadowing tf_env's 3.11.7)
