"""Day 9-10: Train a student variant (V0..V6) on Jester w/ TA per-frame KD.

Single-GPU TensorFlow/Keras training (per Plan: student small, DDP overhead
> compute gain; one variant per 1-GPU partial-node SLURM job).

Pipeline:
    * Input:   Pipeline L (64x64 grayscale, T=8 by default; 48x48 T=6 for V1;
               96x96 T=8 for V6)
    * Backbone: build_clip_model from student_model.py (TimeDistributed Conv
               blocks + causal_tsm_clip between them)
    * Loss:    per-frame weighted KD against TA per-frame logits
               (soft_labels_ta_train.npy). Hard CE + KL with T=4, alpha=0.3,
               uniform frame weights [0.25]*4 at obs ratios {0.25,0.5,0.75,1.0}
    * Optim:   AdamW lr=1e-3 wd=1e-4 + linear warmup + cosine decay
    * Val:     no soft labels — top-1 at obs=1.0 (last student frame logits)
               + per-obs-ratio acc breakdown

Outputs (per run, under MODELS_ROOT/student_{variant}/{job_id}/):
    best.h5             — clip-form Keras weights at best val_obs_1.0
    last.h5             — clip-form Keras weights at last epoch
    metrics.json        — per-epoch train/val metrics + final report
    config.json         — full hyperparams + variant config used

Final metrics row is appended to MODELS_ROOT/student_comparison.csv (created
on first variant; subsequent variants append). Day 10 collect step is in
scripts/analyze_students.py (separate file).
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

from data_loader_tf import build_tf_dataset
from jester_common import parse_split
from kd_dataset_tf import build_kd_dataset
from kd_loss_tf import KDConfig, per_frame_kd_loss, video_level_kd_loss
from path_c_registry import build_clip_model as build_any_clip
from student_model import VARIANTS, build_variant, student_frame_indices


# --------------------------- helpers ---------------------------

def make_lr_schedule(base_lr: float, total_steps: int, warmup_steps: int):
    """Linear warmup -> cosine decay to 0."""

    class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warm = tf.cast(warmup_steps, tf.float32)
            total = tf.cast(total_steps, tf.float32)
            warm_lr = base_lr * step / tf.maximum(warm, 1.0)
            # cosine over (warmup .. total)
            progress = (step - warm) / tf.maximum(total - warm, 1.0)
            cos_lr = 0.5 * base_lr * (1.0 + tf.cos(math.pi * tf.clip_by_value(progress, 0.0, 1.0)))
            return tf.where(step < warm, warm_lr, cos_lr)

        def get_config(self):
            return {"base_lr": base_lr, "total_steps": total_steps, "warmup_steps": warmup_steps}

    return WarmupCosine()


def evaluate(clip_model, val_ds, obs_ratios, T_s: int, max_batches: int = 0,
             video_level: bool = False):
    """Top-1 accuracy. Eager-mode (no @tf.function).

    video_level=False (per-frame): clip_model outputs (B, T_s, K); we score
    at each observation ratio's frame index.
    video_level=True (C2 stacked): clip_model outputs (B, K) directly; only
    obs=1.0 makes sense.
    """
    if video_level:
        n_correct = 0
        n_total = 0
        n_batches = 0
        for clips, labels in val_ds:
            logits = clip_model(clips, training=False)         # (B, K)
            labels_np = labels.numpy() if hasattr(labels, "numpy") else labels
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy()
            n_correct += int((preds == labels_np).sum())
            n_total += len(labels_np)
            n_batches += 1
            if max_batches and n_batches >= max_batches:
                break
        out = {"obs_1.00": float(n_correct / max(n_total, 1)),
               "n_val_samples": n_total, "n_val_batches": n_batches}
        return out

    fi = [max(int(round(T_s * r)) - 1, 0) for r in obs_ratios]
    n_correct = [0] * len(fi)
    n_total = 0
    n_batches = 0
    for clips, labels in val_ds:
        logits = clip_model(clips, training=False)             # (B, T_s, K)
        labels_np = labels.numpy() if hasattr(labels, "numpy") else labels
        for k, idx in enumerate(fi):
            preds = tf.argmax(logits[:, idx], axis=-1, output_type=tf.int32).numpy()
            n_correct[k] += int((preds == labels_np).sum())
        n_total += len(labels_np)
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break
    out = {}
    for k, r in enumerate(obs_ratios):
        out[f"obs_{r:.2f}"] = float(n_correct[k] / max(n_total, 1))
    out["n_val_samples"] = n_total
    out["n_val_batches"] = n_batches
    return out


# --------------------------- training ---------------------------

def main(args):
    tf.get_logger().setLevel("ERROR")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.keras.utils.set_random_seed(args.seed)

    # GPUs: enable memory growth BEFORE any TF op that touches the device
    # (Day 10 bug: building model first initialized the context and made
    # set_memory_growth raise "Physical devices cannot be modified after
    # being initialized").
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"GPUs visible: {len(gpus)}  (this run uses 1)")

    # Now safe to build the model
    clip_model, spec, _stream, _extras = build_any_clip(args.variant)
    out_dir = Path(args.out_dir) / f"student_{spec.name}" / args.job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = int(sum(int(np.prod(w.shape)) for w in clip_model.trainable_weights))
    print(f"=== Training student {spec.name} (job {args.job_id}) -> {out_dir} ===")
    print(f"variant: family={spec.family} mode={spec.input_mode} kd={spec.kd_mode} "
          f"T={spec.T} img={spec.img_size}  params={n_params:,}")

    # Datasets
    n_train = len(parse_split("train"))
    n_val = len(parse_split("validation"))
    full_steps_per_epoch = n_train // args.batch_size
    steps_per_epoch = (
        min(args.max_steps, full_steps_per_epoch) if args.max_steps else full_steps_per_epoch
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    print(f"n_train={n_train}  n_val={n_val}  steps/epoch={steps_per_epoch}  "
          f"total_steps={total_steps}  warmup_steps={warmup_steps}")

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
        mode=spec.input_mode,                            # 'sequence' or 'stacked'
        ta_reduce="last" if spec.kd_mode == "video_level" else None,
    )

    # KD config (only for per-frame mode)
    if spec.kd_mode == "per_frame":
        # legacy obs_ratios; for C1/C3 default to T=8 ratios
        if hasattr(_extras, "get") and _extras.get("cfg") is not None and \
           hasattr(_extras["cfg"], "obs_ratios"):
            obs_ratios = _extras["cfg"].obs_ratios
        else:
            obs_ratios = (0.25, 0.5, 0.75, 1.0)
        s_idx = tuple(student_frame_indices(spec.T, obs_ratios))
        t_idx = tuple(student_frame_indices(8, obs_ratios))
        kd_cfg = KDConfig(
            alpha=args.alpha, temperature=args.temperature,
            student_frame_idx=s_idx, ta_frame_idx=t_idx,
            frame_weights=tuple(args.frame_weights),
        )
        print(f"KD per-frame: alpha={kd_cfg.alpha} T={kd_cfg.temperature} "
              f"s_frames={s_idx} ta_frames={t_idx} weights={kd_cfg.frame_weights}")
    else:
        obs_ratios = (1.0,)        # C2: video-level, only obs=1.0 makes sense
        kd_cfg = None
        print(f"KD video-level: alpha={args.alpha} T={args.temperature} ta_reduce=last")

    # Optimizer (AdamW + warmup-cosine)
    lr_schedule = make_lr_schedule(args.lr, total_steps, warmup_steps)
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=args.wd,
        )
    except AttributeError:
        # TF 2.15: AdamW lives under experimental in some builds
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr_schedule, weight_decay=args.wd,
        )

    if spec.kd_mode == "per_frame":
        @tf.function
        def train_step(clips, labels, soft_logits):
            with tf.GradientTape() as tape:
                logits = clip_model(clips, training=True)
                losses = per_frame_kd_loss(logits, soft_logits, labels, kd_cfg)
                loss = losses["loss"]
            grads = tape.gradient(loss, clip_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, clip_model.trainable_variables))
            return loss, losses["ce"], losses["kd"]
    else:
        # video-level KD (C2 stacked-frame)
        alpha_const = float(args.alpha)
        temp_const = float(args.temperature)
        @tf.function
        def train_step(clips, labels, soft_logits):
            with tf.GradientTape() as tape:
                logits = clip_model(clips, training=True)
                losses = video_level_kd_loss(
                    logits, tf.expand_dims(soft_logits, axis=1),
                    labels, alpha=alpha_const, temperature=temp_const,
                    ta_reduce="last",
                )
                loss = losses["loss"]
            grads = tape.gradient(loss, clip_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, clip_model.trainable_variables))
            return loss, losses["ce"], losses["kd"]

    # Persist config
    config_dump = {
        "variant": spec.name, "family": spec.family,
        "input_mode": spec.input_mode, "kd_mode": spec.kd_mode,
        "T": spec.T, "img_size": spec.img_size, "n_params": n_params,
        "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size, "lr": args.lr, "wd": args.wd,
        "alpha": args.alpha, "temperature": args.temperature,
        "frame_weights": list(args.frame_weights),
        "obs_ratios": list(obs_ratios),
        "soft_labels_ta": str(args.soft_labels_ta),
        "seed": args.seed, "job_id": args.job_id,
    }
    (out_dir / "config.json").write_text(json.dumps(config_dump, indent=2))

    # ---- training loop ----
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
                print(
                    f"  epoch {epoch:02d} step {step:05d}/{steps_per_epoch} "
                    f"loss={loss.numpy():.4f} ce={ce.numpy():.4f} kd={kd.numpy():.4f} "
                    f"lr={cur_lr:.2e}",
                    flush=True,
                )
            if args.max_steps and step >= args.max_steps:
                break
        train_loss = loss_acc / max(n_steps, 1)
        train_ce = ce_acc / max(n_steps, 1)
        train_kd = kd_acc / max(n_steps, 1)
        epoch_train_t = time.time() - t_epoch

        # Validation cadence (Day 9 pipeline is HDF5-bound; full val ~20 min):
        #   * val_every == 0  -> skip ALL val (use scripts/eval_students.py post-hoc)
        #   * else last epoch -> full val (the reported number)
        #   * else if (epoch+1) % val_every == 0 -> fast val
        #   * else skip val for this epoch (still save last.weights.h5)
        is_last_epoch = (epoch == args.epochs - 1)
        do_full_val = args.val_every > 0 and is_last_epoch
        do_fast_val = args.val_every > 0 and (not is_last_epoch) and \
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
            val_metrics = evaluate(clip_model, val_ds, obs_ratios,
                                   T_s=spec.T if spec.kd_mode == "per_frame" else 1,
                                   max_batches=mb,
                                   video_level=(spec.kd_mode == "video_level"))
            del val_ds
        else:
            val_metrics = {"skipped": True}
        epoch_val_t = time.time() - t_val
        val_obs_1 = val_metrics.get("obs_1.00", -1.0)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss, "train_ce": train_ce, "train_kd": train_kd,
            "train_t_s": round(epoch_train_t, 1),
            "val_t_s": round(epoch_val_t, 1),
            **val_metrics,
        }
        history.append(entry)
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
            f"val_obs_1.00={val_obs_1:.4f}  "
            f"val_obs_0.25={val_metrics.get('obs_0.25', 0):.4f} "
            f"val_obs_0.50={val_metrics.get('obs_0.50', 0):.4f} "
            f"val_obs_0.75={val_metrics.get('obs_0.75', 0):.4f}  "
            f"({entry['train_t_s']}s train + {entry['val_t_s']}s val)",
            flush=True,
        )

        # Save best (only when val ran). Skipped-val epochs only update 'last'.
        if val_obs_1 > best_acc and val_obs_1 >= 0.0:
            best_acc = val_obs_1
            clip_model.save_weights(str(out_dir / "best.weights.h5"))
            print(f"  -> new best val_obs_1.00={best_acc:.4f}, saved best.weights.h5")
        # Save last (cheap)
        clip_model.save_weights(str(out_dir / "last.weights.h5"))

        # Append metrics after every epoch
        (out_dir / "metrics.json").write_text(json.dumps({
            "history": history, "best_val_obs_1.00": best_acc,
            "config": config_dump,
        }, indent=2))

    total_t = time.time() - t_total
    print(f"\n=== {spec.name} DONE in {total_t/60:.1f} min   best val_obs_1.00 = {best_acc:.4f} ===")

    csv_path = Path(args.out_dir) / "student_comparison.csv"
    header = (
        "variant,family,mode,kd,params,T,img,"
        "epochs,val_obs_1.00,val_obs_0.75,val_obs_0.50,val_obs_0.25,"
        "train_min,job_id\n"
    )
    last = history[-1] if history else {}
    row = (
        f"{spec.name},{spec.family},{spec.input_mode},{spec.kd_mode},"
        f"{n_params},{spec.T},{spec.img_size},"
        f"{args.epochs},{best_acc:.4f},"
        f"{last.get('obs_0.75', 0):.4f},{last.get('obs_0.50', 0):.4f},{last.get('obs_0.25', 0):.4f},"
        f"{total_t/60:.1f},{args.job_id}\n"
    )
    write_header = not csv_path.exists()
    with csv_path.open("a") as f:
        if write_header:
            f.write(header)
        f.write(row)
    print(f"appended row to {csv_path}")


# --------------------------- CLI ---------------------------

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    help="V0..V9 (legacy), C1a/b/c, C2a/b/c, C3a/b/c (Path C)")
    ap.add_argument("--soft_labels_ta",
                    default=str(PROJECT_ROOT / "soft_labels/soft_labels_ta_train.npy"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--job_id", default=str(int(time.time())))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--fast_val_batches", type=int, default=10,
                    help="# val batches for per-epoch fast monitoring "
                         "(0 = always full val). Last epoch always runs full val.")
    ap.add_argument("--val_every", type=int, default=2,
                    help="Run fast val every N epochs (0 = never). Last epoch "
                         "always runs full val regardless of this setting.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--frame_weights", type=float, nargs=4,
                    default=[0.25, 0.25, 0.25, 0.25])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=0,
                    help=">0 caps steps/epoch (smoke test)")
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
