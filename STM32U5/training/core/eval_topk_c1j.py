"""Quick top-5 (and top-1 sanity) eval for C1j FP32 on Jester val.

Uses the same sync H5 reader as eval_direct.py to avoid the tf.data deadlocks.
"""
from __future__ import annotations

import io
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
from path_c_registry import build_clip_model

CKPT = PROJECT_ROOT / "models/student_C1J/41478885_6/best.weights.h5"


def load_clip_sync(h5, vid, T, img_size):
    ds = h5["clips"][str(vid)]
    offsets = sample_segment_indices(ds.shape[0], T, training=False)
    all_bytes = ds[:]
    first = Image.open(io.BytesIO(bytes(all_bytes[int(offsets[0])]))).convert("L")
    w, h = first.size
    side = min(w, h); top = (h - side) // 2; left = (w - side) // 2
    out = np.empty((T, img_size, img_size, 1), dtype=np.float32)
    for k, off in enumerate(offsets.tolist()):
        img = Image.open(io.BytesIO(bytes(all_bytes[int(off)]))).convert("L")
        img = img.crop((left, top, left + side, top + side)).resize(
            (img_size, img_size), Image.BILINEAR
        )
        out[k, ..., 0] = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return out


def main():
    tf.get_logger().setLevel("ERROR")
    clip_model, spec, _, _ = build_clip_model("C1j")
    clip_model.load_weights(str(CKPT))
    T, img_size = spec.T, spec.img_size
    print(f"C1j FP32 ckpt={CKPT}  T={T} img={img_size}")

    items = parse_split("validation")
    batch_size = 64
    fi_full = T - 1  # obs=1.0 = last student frame index

    n_total = 0
    n_top1 = 0
    n_top5 = 0
    t0 = time.time()
    with h5py.File(str(H5_PATH), "r") as h5:
        batch_clips, batch_labels = [], []
        for ix, (vid, lab) in enumerate(items):
            batch_clips.append(load_clip_sync(h5, vid, T, img_size))
            batch_labels.append(int(lab))
            if len(batch_clips) == batch_size:
                clips = np.stack(batch_clips, axis=0)
                logits = clip_model(clips, training=False).numpy()       # (B, T, 27)
                last_logits = logits[:, fi_full, :]                       # (B, 27) obs=1.0
                top5 = np.argsort(-last_logits, axis=-1)[:, :5]           # descending
                labels = np.asarray(batch_labels, dtype=np.int64)
                n_top1 += int((top5[:, 0] == labels).sum())
                n_top5 += int((top5 == labels[:, None]).any(axis=-1).sum())
                n_total += len(labels)
                batch_clips.clear(); batch_labels.clear()
                if ix % (batch_size * 20) == 0:
                    rate = n_total / (time.time() - t0)
                    print(f"  step {ix:5d}/{len(items)}  top1={n_top1/n_total:.4f}  "
                          f"top5={n_top5/n_total:.4f}  rate={rate:.0f}/s", flush=True)
        # flush remainder
        if batch_clips:
            clips = np.stack(batch_clips, axis=0)
            logits = clip_model(clips, training=False).numpy()
            last_logits = logits[:, fi_full, :]
            top5 = np.argsort(-last_logits, axis=-1)[:, :5]
            labels = np.asarray(batch_labels, dtype=np.int64)
            n_top1 += int((top5[:, 0] == labels).sum())
            n_top5 += int((top5 == labels[:, None]).any(axis=-1).sum())
            n_total += len(labels)

    print(f"\nC1j FP32 final on {n_total} val samples:")
    print(f"  top-1 acc = {n_top1/n_total:.4f}")
    print(f"  top-5 acc = {n_top5/n_total:.4f}")
    print(f"  elapsed   = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
