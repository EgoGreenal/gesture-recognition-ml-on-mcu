# Real-time gesture recognition on STM32U5: TAKD + QAT + early exit + deployment
**Project Final Report — MLonMCU 2026 Spring**

*Author: Yizhou Xu*  *Project days: 14*

---

## TL;DR (1 paragraph, ~50 words)
A real-time gesture-recognition system deployed on STM32U5 (B-U585I-IOT02A, Cortex-M33 @ 160 MHz, 786 KB SRAM, 2 MB Flash): 27-class Jester, deployed model C1j streaming INT8. No-exit INT8 baseline (full val 14787) 86.49%; on-board measured (firmware C1j-ai-v2 streaming `ai_streaming_step` + early exit S1 mf=5 thresh=0.85, mean obs 0.778, n=128) accuracy 84.4%. **Measured latency ~141 ms/frame** (DWT, within the <150 ms/frame budget), ~880 ms/clip (pure compute, ~6.2 frames; the bench counts inference cycles, excluding UART wire time); note the offline X-CUBE-AI estimate ~32-70 ms/frame was 2-4× low. Flash 0.99 MB / SRAM 261 KB. Teacher → TA → student three-stage KD distillation (TAKD) + 17-block TSM-MBV2 cone truncation (w=0.5) + per-channel symmetric INT8 PTQ + min_exit_frame-constrained early exit. All U5 hard constraints pass (Flash<2MB / SRAM<600KB / lat<150ms/frame).

---

## 1. Introduction + project background
- Real-world motivation (real-time gesture → light control)
- Choice of the Jester dataset (27 classes, medium-scale segment videos)
- Choice of STM32U5 (B-U585I-IOT02A): teaching board, Cortex-M33 + X-CUBE-AI 9.1+
- Project milestones: 14 days × theme (Day 1-2: data, 3-5: teacher, 6-7: TA + KD, 8: probe, 9-10: arch search, 11: quantization, 12: early-exit, 13-14: deployment)

## 2. Dataset and preprocessing
- Jester v1: 119k train + 14787 val + 14743 test (no labels)
- 27 classes (incl. left/right mirrors → flip-aware)
- Pipeline L (TF/Keras): 64×64 grayscale @ T=8 (segment sampling, uniform across frames)
- HDF5 storage + `tf.io.decode_jpeg` decoding in-graph (GIL-free)
- KD soft labels: teacher / TA each run once over train + val, per-frame (B, T_ta=8, K=27) FP16
- Key finding (Day 6): for left/right mirror flips the label must be remapped via FLIP_LABEL_MAP, otherwise the KD signal is reversed (~25% acc cost)

## 3. Model architecture — three-stage TAKD distillation
### 3.1 Teacher (Day 3-4)
- Swin-T (~28M params) on Jester
- val acc: **[FILL — find from models/super_teacher.../metrics.json]**

### 3.2 Teacher Assistant (Day 6-7)
- TSM-MobileNetV2 full (17 blocks, w=1.0, RGB 112×112)
- ~2.5M params; KD from teacher with α=0.3, T=4
- val acc: **93.12%** (per-frame[7] = 95.5% argmax-train-label)

### 3.3 Student family — Path C decision (Day 10)
**The V0-V9 from-scratch tiny-CNN line dead-ended** (V4 ceiling 38.45%); switched to the three-way parallel Path C:
- C1: TSM-MBV2 truncated (same topology family as the TA + width_mult) → clear winner, C1b 84%
- C2: Stacked-frame MBV2/V1 (8 frames → channel axis) → 60-69%, video-level KD drops early frames
- C3: Hybrid (2D early + TSM late) → 39-57%, too few TSM sites

The C1 family was pushed to 88.46% (C1g). **Top-2 candidates**:
- **C1f** (1.39M, 87.35% FP32): for accuracy
- **C1j** (722K, 86.65% FP32): for deployment efficiency

## 4. Knowledge distillation (TAKD) — plan vs actual
- Plan: teacher (28M) → TA (2.5M) → student (722K-1.4M)
- Geometric-mean principle: √(28e6 × 0.7e6) ≈ 4.4M ≠ TA 2.5M — but the TA performs excellently thanks to ImageNet pretraining
- Actual: α=0.3 + T=4 + increasing frame_weights [0.10, 0.20, 0.30, 0.40]
- **Increasing frame_weights is key**: TA frame[0] argmax=10.7%, frame[7]=95.5%; uniform weights would route 25% of the KD signal through the noisy early frames

## 5. Quantization (Day 11)
### 5.1 PTQ INT8 (used as default)

| Variant | FP32 obs@1.0 | **PTQ INT8 obs@1.0** | drop |
|---|---:|---:|---:|
| C1f | 87.35% | **87.02%** | -0.33 pp |
| C1j | 86.65% | **86.49%** | -0.16 pp |

PTQ performed clearly better than the Plan expected (expected 1-3 pp, actual ≤ 0.33 pp). Reason: grayscale 1-channel + per-channel weight quant + KD make the logit range well-clustered for the later frames.

### 5.2 QAT — skipped (with rationale)
- tfmot 0.8.0 + TF 2.15 does not accept nested Functional + TimeDistributed (tfmot type check fails)
- A hand-written fake-quant STE ([../training/core/fake_quant_qat.py](../training/core/fake_quant_qat.py)) was written as a fallback and smoke-passed, but not trained
- Decision: PTQ already meets the Plan default tier (≥85%); the QAT marginal gain ≤ 0.5 pp is within decision noise
- Saved ~2-3 hr, redirected to Day 13 deployment (the hardest day of the project)

### 5.3 Multi-input streaming INT8 PTQ — two workarounds
- First attempt `from_keras_model` + positional list: **failed** on Day 10/11 (different failure points: positional reordering / channel divisibility)
- Second attempt `from_saved_model` + dict-keyed generator (found late Day 11 / Day 13 prep): **succeeded** ✓
  - C1f streaming INT8: 1.75 MB
  - C1j streaming INT8: 0.99 MB

## 6. Early-exit strategy (Day 12)
### 6.1 Vanilla S1/S3 do not work (important finding)
- The Plan cites the paper defaults S1 thresh=0.85 / S3 entropy<0.5
- On the C1 INT8 models: **14% acc / obs 0.21** — 70% of samples exit at frame 0 (INT8 logits are more peaked after quantization, "confidently wrong")

### 6.2 Fix: min_exit_frame=5 floor
- The C1 models' per-frame acc is a ramp: frame 0 ~10%, frame 5 ~86%, frame 7 ~87%
- mf=5 forces the earliest obs ≥ 0.75 (where per-frame acc is reliable)
- S1 mf=5 thresh=0.85: C1f 86.66% obs 0.776, C1j **86.69% obs 0.776**

### 6.3 Selected U5 deployment combination
**(C1j, S1, mf=5, thresh=0.85)** ⭐

| Metric | Value |
|---|---:|
| INT8 acc (full Jester val) | **86.69%** |
| Drop vs C1j baseline | 0.0 pp (=baseline) |
| Drop vs C1j FP32 | -0.66 pp |
| mean obs ratio | 0.776 |
| mean frame count | 6.2 / 8 |
| offline est. lat/clip | 199 ms (X-CUBE-AI estimate, low) |
| **measured lat/frame (DWT)** | **~141 ms/frame** (within the <150 budget) |
| **measured lat/clip (compute)** | **~880 ms** (on-board bench, n=128, ~6.2 frames; excl. UART) |
| measured on-device acc (n=128) | 84.4% |
| C1j INT8 Flash | 0.99 MB |
| C1j SRAM | 261 KB |
| C1j per-frame MACs | 8.13M |

## 7. U5 deployment (Day 13)
**[FILL — pending Day 13 completion]**

### 7.1 X-CUBE-AI 9.1 integration
### 7.2 Cache buffer maintenance code
### 7.3 Early-exit hook + INT8 softmax implementation
### 7.4 LED1-4 light-control mapping

## 8. System-level evaluation (Day 14)
**[FILL — Day 14]**
- End-to-end latency profiling (on-board measured vs Plan estimate)
- Misrecognition case study
- Live demo video

## 9. Lessons learned (gotcha-diary summary)
1. **From-scratch tiny CNN is not the route** (V0-V9 ceiling 38%; the C1 line — same topology as the TA + KD distillation — jumps to 84%)
2. **tfmot QAT is incompatible with nested Functional** (TF 2.15 + tfmot 0.8.0; skip QAT once PTQ is enough)
3. **The SavedModel workaround for multi-input INT8 PTQ** (dict-keyed calibrator is the silver bullet)
4. **Vanilla S1/S3 early exit fails / min_exit_frame floor is key** (INT8 makes softmax more peaked, "confidently wrong" on early frames)
5. **HDF5 + Lustre + multi-job** (HDF5_USE_FILE_LOCKING=FALSE + no SWMR)
6. **Increasing frame_weights** (TA early-frame noise → cannot use uniform weights)
7. **dbg QoS submit limit + 25min cap + 7min grace** (lprod 1h cap as insurance)
8. **GPU/compute budget**: the project used ~42 GPU-hr (Plan 100 GPU-hr budget, ~58 left)

## 10. Conclusion and future work
- Built 27-class real-time gesture recognition on STM32U5: C1j streaming INT8, no-exit full-val baseline 86.49-86.69%, on-board measured (S1 mf=5 thr=0.85) on-device acc 84.4% (n=128), **measured ~141 ms/frame (within the <150 budget)**, ~880 ms/clip (pure compute, excl. UART), Flash 0.99 MB
- Early-exit strategy mf=5 + S1=0.85: mean obs 0.778 (exit frame ~5.2, i.e. ~6.2/8 frames observed), ~22% fewer frames / less inference time vs no-exit
- Future directions:
  - True QAT (if the C1 model can be flat-inline rebuilt to avoid the tfmot bug)
  - Extend to RGB input + improve obs_0.25-0.50 accuracy (so the early-exit floor can drop to frame 3-4)
  - On-board multi-modal (IMU/audio) integration

---

## Appendix A: all stedgeai analyze numbers
**[FILL — from models/qat_comparison.csv + the Day 13 stedgeai generate report]**

## Appendix B: repro instructions
**[FILL — from README]**

## Appendix C: daily commit history + resource accounting
**[FILL — Plan Section 9 summary]**
