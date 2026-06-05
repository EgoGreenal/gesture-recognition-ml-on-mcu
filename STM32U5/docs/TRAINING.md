# TRAINING — STM32U5 Track

Three-stage **Teacher-Assistant Knowledge Distillation (TAKD)** + multi-family
student search + QAT/PTQ, targeting INT8 deployment on the B-U585I-IOT02A.

```
Super Teacher  Video Swin-T (28M)   PyTorch, 112x112 RGB   ─┐ video-level soft labels
      │ KD                                                   │
Teacher Assistant  TSM-MobileNetV2 (3.5M, uni-directional)  ─┤ per-frame soft labels (.npy)
      │ KD                                                   │
Student  Path-C families (C1/C2/C3)  TF/Keras, 64x64 gray  ─┘
      │ QAT / PTQ → INT8 TFLite
      ▼
ST Edge AI Core → C code → STM32U5
```

Framework bridge: PyTorch teachers export soft labels as `.npy`; the TF/Keras
student reads them directly. No weight conversion.

## 0. Prerequisite

Environment + dataset as in [INSTALL.md](INSTALL.md). Activate the right venv per
stage (`torch` for stages 1-2, `tf` for stages 3-4).

## 1. Data

- `training/core/tgz_to_hdf5.py` — stream the Jester tgz into one HDF5 file.
- `training/core/data_loader_torch.py` — Pipeline H (112x112 RGB, 8 frames) for
  Teacher/TA.
- `training/core/data_loader_tf.py` — Pipeline L (64x64 grayscale, 8 frames) for
  the student; supports `mode="sequence"` (C1/C3) and `mode="stacked"` (C2).
- `training/core/verify_jester.py` — integrity / split-alignment check.

## 2. Stage 1 — Super Teacher (Video Swin-T)

`training/core/train_super_teacher.py` — 1 node × 4 GPU DDP, AdamW, cosine
schedule. Outputs `super_teacher_swin_t.pth` and **video-level** soft labels
(`soft_labels_train*.npy`). Swin-T is bi-directional → video-level only; it
supervises the TA, not the student.

`training/core/swa_tta_eval.py` provides the SWA+TTA evaluation used to pick the
strongest teacher checkpoint for distillation.

## 3. Stage 2 — Teacher Assistant (TSM-MobileNetV2)

`training/core/train_ta.py` + `ta_tsm_mbv2.py`. **Uni-directional** shift
(`shift_div=8`), head modified to emit **per-frame** logits `[B, T=8, 27]`
(no mean-over-T) so frame-t predictions are causal. Distilled from the Swin-T
soft labels. Exports **per-frame** soft labels via
`training/core/export_ta_softlabels.py` → `soft_labels_ta_train.npy`
(`[N_train, 8, 27]`, FP16). This is the student's KD target.

## 4. Stage 3 — Student search (Path-C)

The from-scratch tiny-CNN line (V0-V6) topped out <40% and was abandoned (see
`../results/model_training_summary.csv`). Path-C explores three families:

| Family | Factory | Input | KD |
|---|---|---|---|
| C1 — TSM-MBV2 truncated | `training/core/path_c_registry.py` / `tsm_mbv2_tf.py` | `[B,8,64,64,1]` sequence | per-frame |
| C2 — stacked-frame MobileNet | `training/core/path_c2_factory.py` | `[B,64,64,8]` stacked | video-level |
| C3 — stacked + late-TSM hybrid | `training/core/path_c3_factory.py` | sequence | per-frame (TSM tail) |

Train one variant:

```bash
python training/core/train_student.py \
    --variant C1j \
    --soft_labels_ta soft_labels/soft_labels_ta_train.npy \
    --out_dir models/ \
    --epochs 30 --batch_size 64 --lr 1e-3 \
    --alpha 0.3 --temperature 4.0
```

KD loss (`training/core/kd_loss_tf.py`): `alpha`·CE(hard) +
(1-`alpha`)·T²·KL(soft) — per-frame for C1/C3 (weighted over frames via
`--frame_weights`), video-level (TA last frame) for C2. All variants as a SLURM
array: `training/server_scripts/sbatch/train_path_c_array.sh`.

**Selection** (`training/core/analyze_path_c.py` → `../results/path_selection/`):
survivors under the 600 KB SRAM / 150 ms-per-frame budgets were **C1f, C1j, C1n,
C1o**. C1f/C1j were carried to quantization. FP32 best top-1 ≈ 86.6–87.4 %.

## 5. Stage 4 — Quantization

- QAT: `training/core/train_qat.py` (+ `fake_quant_qat.py`) — wrap with
  `tfmot.quantization.keras.quantize_model`, fine-tune 5 ep at lr 1e-5, keep KD on.
- PTQ baseline: `training/core/export_int8.py` with a ~500-sample representative set.
- Streaming export (single-frame + cache I/O for real-time early exit) vs clip
  export: `tsm_streaming_layer.py`, `tsm_probe.py` (see
  [../docs/tsm_probe_results.md](tsm_probe_results.md)).

Host INT8 evaluation (`training/core/eval_int8.py` → `../results/int8_eval/`):
**C1f 87.02 %**, **C1j 86.49 %** top-1 on the 14,787-clip val set.

## 6. Early exit

See [EARLY_EXIT.md](EARLY_EXIT.md).

## Reproducing the result summaries

`../results/{confusion_summary,model_training_summary,model_phase_history_summary}.csv`
are derived from training logs + result JSONs + saved val logits — **no retraining
needed**. See `../results/README.md`.
