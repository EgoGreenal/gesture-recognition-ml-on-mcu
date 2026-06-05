"""Day 11: QAT fine-tune for C1 Top-2 (C1f / C1j) starting from FP32 ckpt.

QAT approach: fake-quant straight-through estimator on Conv2D / DepthwiseConv2D
/ Dense kernels (symmetric per-output-channel int8). See `fake_quant_qat.py`
for the math. tfmot.quantization.keras.quantize_model() does NOT work on our
nested-Functional architecture in tfmot 0.8.0 + TF 2.15 (raises
"to_annotate can only be a keras.Model instance" -- the C1 clip_model is a
Functional with TimeDistributed(Functional submodel) nesting that tfmot's
type-check rejects). The manual fake-quant approach side-steps this entirely.

Training mirrors the FP32 path (train_student.py) but:
  - lr = 1e-5 instead of 1e-3 (light fine-tune)
  - 5 epochs (down from 30-60)
  - warmup_epochs = 0 (already warm)
  - load_fp32_weights = best.weights.h5 from Day 10
  - install_fake_quant_on_model() patches Conv/Dense to fake-quant on forward
  - same KD loss + TA per-frame soft labels (per Plan: "keep using TA per-frame soft labels for KD")
  - same frame_weights [0.10, 0.20, 0.30, 0.40] + alpha=0.3 + T=4.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data_loader_tf import build_tf_dataset, build_kd_dataset
from fake_quant_qat import install_fake_quant_on_model, uninstall_fake_quant
from jester_common import parse_split
from kd_loss_tf import KDConfig, per_frame_kd_loss
from path_c_registry import build_clip_model
from student_model import student_frame_indices


def make_lr_schedule(base_lr: float, total_steps: int, warmup_steps: int):
    class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warm = tf.cast(warmup_steps, tf.float32)
            total = tf.cast(total_steps, tf.float32)
            warm_lr = base_lr * step / tf.maximum(warm, 1.0)
            progress = (step - warm) / tf.maximum(total - warm, 1.0)
            cos_lr = 0.5 * base_lr * (1.0 + tf.cos(math.pi * tf.clip_by_value(progress, 0.0, 1.0)))
            return tf.where(step < warm, warm_lr, cos_lr)

        def get_config(self):
            return {"base_lr": base_lr, "total_steps": total_steps, "warmup_steps": warmup_steps}

    return WarmupCosine()


def evaluate(clip_model, val_ds, obs_ratios, T_s: int, max_batches: int = 0):
    fi = [max(int(round(T_s * r)) - 1, 0) for r in obs_ratios]
    n_correct = [0] * len(fi)
    n_total = 0
    n_batches = 0
    for clips, labels in val_ds:
        logits = clip_model(clips, training=False)
        labels_np = labels.numpy() if hasattr(labels, "numpy") else labels
        for k, idx in enumerate(fi):
            preds = tf.argmax(logits[:, idx], axis=-1, output_type=tf.int32).numpy()
            n_correct[k] += int((preds == labels_np).sum())
        n_total += len(labels_np)
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break
    return {f"obs_{r:.2f}": float(n_correct[k] / max(n_total, 1))
            for k, r in enumerate(obs_ratios)} | {
        "n_val_samples": n_total, "n_val_batches": n_batches,
    }


def main(args):
    tf.get_logger().setLevel("ERROR")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.keras.utils.set_random_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"GPUs visible: {len(gpus)}", flush=True)

    # Build FP32 clip_model + load Day 10 weights
    clip_model, spec, _stream, _extras = build_clip_model(args.variant)
    out_dir = Path(args.out_dir) / f"student_{spec.name}" / "qat" / args.job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = int(sum(int(np.prod(w.shape)) for w in clip_model.trainable_weights))
    print(f"=== QAT {spec.name} (job {args.job_id}) -> {out_dir} ===")
    print(f"  family={spec.family} T={spec.T} img={spec.img_size} params={n_params:,}")
    print(f"  loading FP32 weights: {args.load_fp32_weights}")
    clip_model.load_weights(args.load_fp32_weights)
    print(f"  FP32 weights loaded")

    # Install fake-quant on Conv2D / DepthwiseConv2D / Dense (recursive)
    n_patched = install_fake_quant_on_model(clip_model, num_bits=8)
    print(f"  installed fake-quant on {n_patched} Conv/Dense layers (8-bit STE)")

    # Datasets
    n_train = len(parse_split("train"))
    full_steps_per_epoch = n_train // args.batch_size
    steps_per_epoch = (
        min(args.max_steps, full_steps_per_epoch) if args.max_steps else full_steps_per_epoch
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    print(f"  n_train={n_train}  steps/ep={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}")

    train_ds = build_kd_dataset(
        split="train",
        batch_size=args.batch_size,
        soft_labels_path=args.soft_labels_ta,
        T=spec.T,
        img_size=spec.img_size,
        training=True,
        num_shards=args.num_workers,
        shuffle_buf=128,
        seed=args.seed,
        mode=spec.input_mode,
    )

    obs_ratios = (0.25, 0.5, 0.75, 1.0)
    s_idx = tuple(student_frame_indices(spec.T, obs_ratios))
    t_idx = tuple(student_frame_indices(8, obs_ratios))
    kd_cfg = KDConfig(
        alpha=args.alpha, temperature=args.temperature,
        student_frame_idx=s_idx, ta_frame_idx=t_idx,
        frame_weights=tuple(args.frame_weights),
    )
    print(f"  KD: alpha={kd_cfg.alpha} T={kd_cfg.temperature} "
          f"s_idx={s_idx} ta_idx={t_idx} weights={kd_cfg.frame_weights}")

    lr_schedule = make_lr_schedule(args.lr, total_steps, warmup_steps)
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=args.wd,
        )
    except AttributeError:
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr_schedule, weight_decay=args.wd,
        )

    @tf.function
    def train_step(clips, labels, soft_logits):
        with tf.GradientTape() as tape:
            logits = clip_model(clips, training=True)        # fake-quant ON
            losses = per_frame_kd_loss(logits, soft_logits, labels, kd_cfg)
            loss = losses["loss"]
        grads = tape.gradient(loss, clip_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, clip_model.trainable_variables))
        return loss, losses["ce"], losses["kd"]

    config_dump = {
        "variant": spec.name, "family": spec.family, "input_mode": spec.input_mode,
        "T": spec.T, "img_size": spec.img_size, "n_params": n_params,
        "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size, "lr": args.lr, "wd": args.wd,
        "alpha": args.alpha, "temperature": args.temperature,
        "frame_weights": list(args.frame_weights),
        "soft_labels_ta": str(args.soft_labels_ta),
        "load_fp32_weights": args.load_fp32_weights,
        "n_fake_quant_layers": n_patched,
        "seed": args.seed, "job_id": args.job_id,
        "qat_approach": "fake_quant_ste_weight_only",
    }
    (out_dir / "config.json").write_text(json.dumps(config_dump, indent=2))

    history = []
    best_acc = -1.0
    t_total = time.time()
    for epoch in range(args.epochs):
        t_epoch = time.time()
        loss_acc = ce_acc = kd_acc = 0.0
        n_steps = 0
        for step, (clips, labels, soft_logits) in enumerate(train_ds):
            loss, ce, kd = train_step(clips, labels, soft_logits)
            loss_acc += float(loss.numpy())
            ce_acc += float(ce.numpy())
            kd_acc += float(kd.numpy())
            n_steps += 1
            if step % args.log_every == 0:
                cur_lr = float(lr_schedule(optimizer.iterations).numpy())
                print(f"  epoch {epoch:02d} step {step:05d}/{steps_per_epoch} "
                      f"loss={loss.numpy():.4f} ce={ce.numpy():.4f} kd={kd.numpy():.4f} "
                      f"lr={cur_lr:.2e}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        train_loss = loss_acc / max(n_steps, 1)
        train_ce = ce_acc / max(n_steps, 1)
        train_kd = kd_acc / max(n_steps, 1)
        epoch_train_t = time.time() - t_epoch

        # Always full val on last epoch; fast val on others if val_every > 0
        is_last = (epoch == args.epochs - 1)
        do_full_val = args.val_every > 0 and is_last
        do_fast_val = args.val_every > 0 and (not is_last) and \
                      ((epoch + 1) % args.val_every == 0)
        t_val = time.time()
        if do_full_val or do_fast_val:
            val_ds = build_tf_dataset(
                split="validation",
                batch_size=args.batch_size,
                T=spec.T,
                img_size=spec.img_size,
                training=False,
                num_shards=2,
                seed=args.seed,
                mode=spec.input_mode,
            )
            mb = 0 if do_full_val else args.fast_val_batches
            val_metrics = evaluate(clip_model, val_ds, obs_ratios, T_s=spec.T,
                                   max_batches=mb)
            del val_ds
        else:
            val_metrics = {"skipped": True}
        epoch_val_t = time.time() - t_val
        val_obs_1 = val_metrics.get("obs_1.00", -1.0)

        entry = {
            "epoch": epoch, "train_loss": train_loss, "train_ce": train_ce,
            "train_kd": train_kd, "train_t_s": round(epoch_train_t, 1),
            "val_t_s": round(epoch_val_t, 1), **val_metrics,
        }
        history.append(entry)
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
              f"val_obs_1.00={val_obs_1:.4f} val_obs_0.75={val_metrics.get('obs_0.75', 0):.4f} "
              f"({entry['train_t_s']}s train + {entry['val_t_s']}s val)", flush=True)

        if val_obs_1 > best_acc and val_obs_1 >= 0.0:
            best_acc = val_obs_1
            clip_model.save_weights(str(out_dir / "best.weights.h5"))
            print(f"  -> new best val_obs_1.00={best_acc:.4f}, saved best.weights.h5", flush=True)
        clip_model.save_weights(str(out_dir / "last.weights.h5"))

        (out_dir / "metrics.json").write_text(json.dumps({
            "history": history, "best_val_obs_1.00": best_acc,
            "config": config_dump,
        }, indent=2))

    total_t = time.time() - t_total
    print(f"\n=== QAT {spec.name} DONE in {total_t/60:.1f} min  "
          f"best val_obs_1.00 = {best_acc:.4f} ===")

    # Save also an "uninstalled" copy so downstream tools can load weights into
    # a fresh FP32 model without needing the fake-quant install hook
    uninstall_fake_quant(clip_model)
    clip_model.save_weights(str(out_dir / "best_clean.weights.h5"))
    print(f"  saved uninstalled best_clean.weights.h5 (FP32 model can load directly)")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--load_fp32_weights", required=True,
                    help="Path to Day 10 FP32 best.weights.h5")
    ap.add_argument("--soft_labels_ta",
                    default=str(PROJECT_ROOT / "soft_labels/soft_labels_ta_train.npy"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--job_id", default=str(int(time.time())))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--warmup_epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--frame_weights", type=float, nargs=4,
                    default=[0.10, 0.20, 0.30, 0.40])
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--fast_val_batches", type=int, default=30)
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=0)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
