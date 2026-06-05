"""Day 10 fallback: synchronous validation, NO tf.data, NO multi-threading.

Day 9-10 lesson: `tf.data.Dataset.from_generator` + h5py + SWMR + concurrent
SLURM jobs reading the same HDF5 file all simultaneously is a recipe for
deadlock. We've seen the eval process accumulate 47+ SLEEPING threads while
waiting on h5py reads that never return.

This script bypasses tf.data entirely:
  1. Open HDF5 once (this process only)
  2. Iterate val items in a plain Python `for` loop
  3. Decode JPEG via PIL (synchronous, single-thread)
  4. Stack into a numpy batch
  5. Call clip_model(batch) -- eager forward pass
  6. Accumulate predictions

Slower per-sample than a properly-functioning tf.data pipeline, but
deterministic and deadlock-free. ~5-10 min per variant for full Jester val.

Usage:
    python eval_direct.py --variants V0 V1 V2 V3 V4 V5 V6
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from jester_common import H5_PATH, parse_split, sample_segment_indices
from student_model import VARIANTS, build_variant
from eval_students import find_latest_run


def load_clip_sync(h5: h5py.File, vid: int, T: int, img_size: int) -> np.ndarray:
    """Synchronous (val) clip loader. No augment, deterministic sampling."""
    ds = h5["clips"][str(vid)]
    n_frames = ds.shape[0]
    offsets = sample_segment_indices(n_frames, T, training=False)
    first = Image.open(io.BytesIO(bytes(ds[int(offsets[0])]))).convert("L")
    w, h = first.size
    side = min(w, h); top = (h - side) // 2; left = (w - side) // 2
    out = np.empty((T, img_size, img_size, 1), dtype=np.float32)
    for k, off in enumerate(offsets.tolist()):
        img = first if k == 0 else Image.open(io.BytesIO(bytes(ds[int(off)]))).convert("L")
        img = img.crop((left, top, left + side, top + side)).resize(
            (img_size, img_size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [-1, 1]
        out[k, ..., 0] = arr
    return out


def eval_one_variant(name: str, models_root: Path, batch_size: int = 64,
                     max_samples: int = 0, prefer: str = "best") -> dict:
    found = find_latest_run(models_root, name, prefer=prefer)
    if found is None:
        return {"variant": name, "status": "no_ckpt"}
    ckpt, meta = found
    cfg = VARIANTS[name]
    print(f"\n=== Eval {name}  ckpt={ckpt}")

    bundle = build_variant(cfg)
    clip_model = bundle["clip_model"]
    clip_model.load_weights(str(ckpt))
    print(f"  loaded weights ({int(sum(int(np.prod(w.shape)) for w in clip_model.trainable_weights))} params)")

    items = parse_split("validation")
    if max_samples and max_samples < len(items):
        items = items[:max_samples]

    # Frame indices at obs ratios
    fi = [max(int(round(cfg.T * r)) - 1, 0) for r in cfg.obs_ratios]

    n_correct = [0] * len(fi)
    n_total = 0
    t0 = time.time()
    batch_clips: list[np.ndarray] = []
    batch_labels: list[int] = []

    # Open HDF5 in plain read-only mode (no libver=latest, no SWMR). Day 10
    # lesson: SWMR + many concurrent SLURM readers via Lustre serializes on
    # file-level locks. Default mode is more forgiving for many-reader-no-writer.
    with h5py.File(str(H5_PATH), "r") as h5:
        for ix, (vid, label) in enumerate(items):
            clip = load_clip_sync(h5, vid, cfg.T, cfg.H)
            batch_clips.append(clip)
            batch_labels.append(int(label))
            if len(batch_clips) == batch_size:
                clips_b = np.stack(batch_clips, axis=0)
                labels_b = np.asarray(batch_labels, dtype=np.int32)
                logits = clip_model(clips_b, training=False).numpy()   # (B, T_s, K)
                for k, idx in enumerate(fi):
                    preds = logits[:, idx].argmax(-1).astype(np.int32)
                    n_correct[k] += int((preds == labels_b).sum())
                n_total += len(batch_labels)
                batch_clips.clear()
                batch_labels.clear()
                if ix % (batch_size * 10) == 0:
                    rate = n_total / (time.time() - t0)
                    print(f"  step {ix:5d}/{len(items)}  "
                          f"acc1.0_so_far={n_correct[-1]/max(n_total,1):.4f}  "
                          f"rate={rate:.1f} samples/s", flush=True)
        # Flush remainder
        if batch_clips:
            clips_b = np.stack(batch_clips, axis=0)
            labels_b = np.asarray(batch_labels, dtype=np.int32)
            logits = clip_model(clips_b, training=False).numpy()
            for k, idx in enumerate(fi):
                preds = logits[:, idx].argmax(-1).astype(np.int32)
                n_correct[k] += int((preds == labels_b).sum())
            n_total += len(batch_labels)

    out = {"variant": name, "status": "ok", "ckpt": str(ckpt),
           "n_val_samples": n_total, "val_time_s": round(time.time() - t0, 1)}
    for k, r in enumerate(cfg.obs_ratios):
        out[f"val_obs_{r:.2f}"] = round(n_correct[k] / max(n_total, 1), 4)

    # Persist eval_full.json next to the run dir (analyze_students.py reads it)
    run_dir = Path(meta["run_dir"])
    (run_dir / "eval_full.json").write_text(json.dumps({
        "ckpt": str(ckpt), "variant": name,
        "metrics": {k.replace("val_obs_", "val_obs_"): v for k, v in out.items()
                    if k.startswith("val_obs_")},
        "n_val_samples": n_total, "val_time_s": out["val_time_s"],
    }, indent=2))
    print(f"  wrote {run_dir/'eval_full.json'}")
    print(f"  V{name[1:]}: acc@1.0={out['val_obs_1.00']:.4f}  acc@0.75={out['val_obs_0.75']:.4f}  "
          f"acc@0.5={out['val_obs_0.50']:.4f}  acc@0.25={out['val_obs_0.25']:.4f}  "
          f"({n_total} samples, {out['val_time_s']}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    ap.add_argument("--models_root", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=0,
                    help="0 = full val; >0 caps samples (for quick sanity check)")
    ap.add_argument("--prefer", choices=["best", "last"], default="best")
    args = ap.parse_args()

    tf.get_logger().setLevel("ERROR")

    rows = []
    for name in args.variants:
        try:
            r = eval_one_variant(name, Path(args.models_root), args.batch_size,
                                 max_samples=args.max_samples, prefer=args.prefer)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            r = {"variant": name, "status": "error", "error": str(e)}
        rows.append(r)

    print("\n=== Eval summary ===")
    print(f"{'V':<4}{'acc@1.0':>9}{'acc@0.75':>9}{'acc@0.50':>9}{'acc@0.25':>9}{'n':>8}{'t_s':>7}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['variant']:<4}  ({r.get('status', '?')})")
            continue
        print(
            f"{r['variant']:<4}{r.get('val_obs_1.00', 0):>9.4f}{r.get('val_obs_0.75', 0):>9.4f}"
            f"{r.get('val_obs_0.50', 0):>9.4f}{r.get('val_obs_0.25', 0):>9.4f}"
            f"{r['n_val_samples']:>8}{r['val_time_s']:>7.1f}"
        )


if __name__ == "__main__":
    main()
