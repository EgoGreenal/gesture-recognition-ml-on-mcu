#!/usr/bin/env python3
"""dump_val_clips_int8.py - Leonardo-side helper.

Reads jester_full.h5 (per Project_Plan_ch.md Day 1 layout), picks the first N
validation clips per the official Jester split, applies the SAME
center-crop -> 64x64 grayscale -> float [-1,1] -> int8 quantization pipeline
as deploy/host_send_jester.py, and writes a portable .npz that
host_benchmark.py (on Windows) can consume directly.

Run on Leonardo (where jester_full.h5 + scripts/jester_common.py live).
NOT for running on the Windows host -- it depends on the cluster paths.

    python dump_val_clips_int8.py --n 100 \\
           --out /leonardo_work/.../jester_val_int8_100.npz

Then scp the .npz down to the Windows project dir:
    scp leonardo:.../jester_val_int8_100.npz \\
        D:/STM32CubeIDE_2.0.0/gesture_c1j_U5_board/jester_val_int8.npz

Resulting npz schema (consumed by host_benchmark.py):
    clips_int8 : (N, 8, 64, 64) int8     # NHWC channel collapsed to 2D
    labels     : (N,)            int32
    vids       : (N,)            int64   # Jester video IDs
    class_names: (27,)           str     # optional, included if available
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


T_FRAMES = 8
IMG_SIZE = 64

# Must match the deployed model's input quantization
# (X-CUBE-AI/App/student_c1j_ptq_int8_generate_report.txt -> input 1/11).
IN_SCALE = 0.007843138
IN_ZP    = -1


def quantize_frame_int8(frame_fp: np.ndarray) -> np.ndarray:
    """frame_fp: (H, W) float in [-1, 1]. Returns (H, W) int8."""
    q = np.round(frame_fp / IN_SCALE + IN_ZP).clip(-128, 127).astype(np.int8)
    return q


def import_jester_common(project_root: Path):
    """Wire up scripts/jester_common.py so we get H5_PATH + helpers."""
    sys.path.insert(0, str(project_root / "scripts"))
    import jester_common  # noqa: F401
    return jester_common


def maybe_load_class_names(project_root: Path) -> list[str] | None:
    """Best-effort load of the 27 Jester class names; returns None if unavailable.

    Standard Jester layout has jester-v1-labels.csv (one class per line, sorted
    alphabetically -> class index). If your H5_PATH parent has it, we'll pick
    it up; otherwise we just omit class_names and host_benchmark.py falls back
    to numeric labels.
    """
    candidates = [
        project_root / "data" / "jester" / "jester-v1-labels.csv",
        project_root / "scripts" / "jester-v1-labels.csv",
        Path("jester-v1-labels.csv"),
    ]
    for c in candidates:
        if c.exists():
            names = [ln.strip() for ln in c.read_text().splitlines() if ln.strip()]
            if len(names) == 27:
                return names
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100,
                    help="number of validation clips to dump (default: 100)")
    ap.add_argument("--out", required=True,
                    help="output .npz path")
    ap.add_argument("--project-root",
                    default="/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU",
                    help="project root containing scripts/jester_common.py "
                         "(default: %(default)s)")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    jc = import_jester_common(project_root)

    items = jc.parse_split("validation")[:args.n]
    n = len(items)
    print(f"loading {n} validation clips from {jc.H5_PATH}")

    clips = np.empty((n, T_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.int8)
    labels = np.empty((n,), dtype=np.int32)
    vids = np.empty((n,), dtype=np.int64)

    with h5py.File(str(jc.H5_PATH), "r", libver="latest", swmr=True) as h5:
        for k, (vid, label) in enumerate(items):
            ds = h5["clips"][str(vid)]
            n_frames = ds.shape[0]
            offsets = jc.sample_segment_indices(n_frames, T_FRAMES, training=False)
            all_bytes = ds[:]
            jpegs = [bytes(all_bytes[int(o)]) for o in offsets.tolist()]

            # Center-square crop based on the first frame's full resolution
            first = Image.open(io.BytesIO(jpegs[0])).convert("L")
            w, h = first.size
            side = min(w, h)
            top = (h - side) // 2
            left = (w - side) // 2

            for t, jpg in enumerate(jpegs):
                img = Image.open(io.BytesIO(jpg)).convert("L")
                img = img.crop((left, top, left + side, top + side)).resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR
                )
                arr_fp = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                clips[k, t] = quantize_frame_int8(arr_fp)

            labels[k] = int(label)
            vids[k] = int(vid)

            if (k + 1) % 25 == 0 or (k + 1) == n:
                print(f"  [{k+1:4d}/{n}] vid={vid} label={label}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(clips_int8=clips, labels=labels, vids=vids)
    names = maybe_load_class_names(project_root)
    if names is not None:
        payload["class_names"] = np.array(names)
        print(f"included class_names ({len(names)})")
    else:
        print("class_names not found -- benchmark will print numeric labels")

    np.savez(out, **payload)
    sz_mb = out.stat().st_size / 1024 / 1024
    print(f"wrote {out}  ({sz_mb:.1f} MiB)")
    print(f"  clips_int8: shape={clips.shape} dtype={clips.dtype} "
          f"min={clips.min()} max={clips.max()}")
    print(f"  labels    : shape={labels.shape} dtype={labels.dtype} "
          f"unique={len(np.unique(labels))}")


if __name__ == "__main__":
    main()
