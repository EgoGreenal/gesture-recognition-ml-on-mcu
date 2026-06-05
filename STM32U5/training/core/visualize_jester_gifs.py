#!/usr/bin/env python3
"""
Create GIF previews from the 20BN-Jester HDF5 file used in the MLonMCU plan.

Typical use on Leonardo:
  python scripts/visualize_jester_gifs.py \
      --h5 data/jester/jester_full.h5 \
      --split_csv data/jester/annotations/jester-v1-validation.csv \
      --labels_txt data/jester/jester_labels.txt \
      --out outputs/jester_gifs \
      --num 8 \
      --view student \
      --frames uniform8
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class SplitItem:
    video_id: str
    label: str | None


def _clean_cell(x: str) -> str:
    return x.strip().strip('"').strip("'")


def read_labels(labels_txt: Path | None) -> list[str]:
    if labels_txt is None or not labels_txt.exists():
        return []
    labels = []
    with labels_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


def parse_split_csv(split_csv: Path, labels: Sequence[str]) -> list[SplitItem]:
    """Parse official Jester split CSVs.

    Supports both semicolon-separated files like `video_id;label name` and simple
    comma-separated variants. Test CSVs may contain only video_id.
    """
    items: list[SplitItem] = []
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = ";" if sample.count(";") >= sample.count(",") else ","
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            row = [_clean_cell(c) for c in row]
            if not row[0] or row[0].lower() in {"video_id", "id"}:
                continue
            video_id = row[0]
            label: str | None = None
            if len(row) >= 2 and row[1] != "":
                raw_label = row[1]
                if raw_label.isdigit() and labels:
                    idx = int(raw_label)
                    label = labels[idx] if 0 <= idx < len(labels) else raw_label
                else:
                    label = raw_label
            items.append(SplitItem(video_id=video_id, label=label))
    return items


def natural_key(s: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s)))


def bytes_from_h5_value(value) -> bytes:
    """Convert a scalar/array HDF5 value that stores JPEG bytes into bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.void):
        return bytes(value)
    arr = np.asarray(value)
    if arr.dtype == np.uint8:
        return arr.tobytes()
    if arr.dtype.kind in {"S", "O", "U"}:
        # vlen bytes sometimes arrive as an object array of uint8 arrays or bytes.
        if arr.shape == ():
            scalar = arr.item()
            if isinstance(scalar, bytes):
                return scalar
            if isinstance(scalar, np.ndarray) and scalar.dtype == np.uint8:
                return scalar.tobytes()
        if len(arr) > 0 and isinstance(arr.flat[0], (bytes, np.ndarray)):
            first = arr.flat[0]
            if isinstance(first, bytes):
                return first
            if isinstance(first, np.ndarray) and first.dtype == np.uint8:
                return first.tobytes()
    raise TypeError(f"Cannot convert HDF5 value with type={type(value)} dtype={getattr(arr, 'dtype', None)} to bytes")


def iter_frame_bytes_from_obj(obj) -> Iterator[bytes]:
    """Yield JPEG bytes from a dataset/group representing one video clip."""
    if isinstance(obj, h5py.Group):
        for key in sorted(obj.keys(), key=natural_key):
            child = obj[key]
            if isinstance(child, h5py.Dataset):
                yield bytes_from_h5_value(child[()])
            else:
                yield from iter_frame_bytes_from_obj(child)
        return

    if isinstance(obj, h5py.Dataset):
        if obj.ndim == 0:
            yield bytes_from_h5_value(obj[()])
        else:
            for i in range(obj.shape[0]):
                yield bytes_from_h5_value(obj[i])
        return

    # Handles an already-indexed dataset row such as h5["frames"][idx].
    arr = np.asarray(obj)
    if arr.dtype == object:
        for x in arr:
            yield bytes_from_h5_value(x)
    elif arr.ndim == 1 and arr.dtype == np.uint8:
        yield arr.tobytes()
    elif arr.ndim >= 1:
        for x in arr:
            yield bytes_from_h5_value(x)
    else:
        yield bytes_from_h5_value(arr)


def decode_video_ids(value) -> list[str]:
    out = []
    for x in value:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def find_clip_obj(h5: h5py.File, video_id: str):
    """Find one clip in a few common HDF5 layouts.

    The converter used in the plan stores per-clip vlen JPEG bytes. Depending on
    implementation details, clips may be stored as root keys, inside a group, or
    in an indexed dataset with a `video_ids` array. This function handles all
    three patterns without requiring a full recursive scan over 148k clips.
    """
    candidates = [
        video_id,
        f"clips/{video_id}",
        f"videos/{video_id}",
        f"frames/{video_id}",
        f"data/{video_id}",
        f"jester/{video_id}",
    ]
    for key in candidates:
        if key in h5:
            return h5[key], key

    # Indexed layout: video_ids + one dataset/group containing all clips.
    for ids_key in ["video_ids", "ids", "clip_ids"]:
        if ids_key in h5:
            ids = decode_video_ids(h5[ids_key][()])
            try:
                idx = ids.index(video_id)
            except ValueError:
                continue
            for data_key in ["clips", "videos", "frames", "data", "jpeg_bytes"]:
                if data_key in h5:
                    data = h5[data_key]
                    if isinstance(data, h5py.Dataset):
                        return data[idx], f"{data_key}[{idx}]"
                    if isinstance(data, h5py.Group):
                        # Group may be keyed by index rather than video_id.
                        for k in [str(idx), str(idx).zfill(6), video_id]:
                            if k in data:
                                return data[k], f"{data_key}/{k}"
            raise KeyError(f"Found video_id {video_id} at index {idx}, but could not find a clip data dataset/group.")

    raise KeyError(
        f"Could not find video_id={video_id!r} in HDF5. Root keys include: {list(h5.keys())[:20]}"
    )


def select_indices(n: int, mode: str, uniform_count: int, max_frames: int) -> list[int]:
    if n <= 0:
        return []
    if mode == "uniform8":
        count = min(uniform_count, n)
        return sorted(set(np.linspace(0, n - 1, count).round().astype(int).tolist()))
    if mode == "segment8":
        # TSN center-of-segment sampling (matches jester_common.sample_segment_indices,
        # training=False). This is the exact sampling early_exit_eval uses to
        # produce the logits, so GIF frame j corresponds to logits row j.
        T = uniform_count
        if n >= T:
            avg = n / float(T)
            offsets = (np.arange(T) * avg + avg / 2.0).astype(int)
            offsets = np.minimum(offsets, n - 1)
        else:
            offsets = np.minimum(np.arange(T), n - 1).astype(int)
        # Keep duplicates if n < T so GIF still has T frames.
        return offsets.tolist()
    if mode == "all":
        if n <= max_frames:
            return list(range(n))
        return sorted(set(np.linspace(0, n - 1, max_frames).round().astype(int).tolist()))
    raise ValueError(f"Unknown frame mode: {mode}")


def _softmax_last(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _entropy_last(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(p + 1e-12), axis=-1)


def decide_exit_frame(probs: np.ndarray, strategy: str, threshold: float,
                      min_exit_frame: int) -> int:
    """Return the index of the first frame whose policy condition fires.

    probs: (T, K) softmax probs. Returns T-1 if no frame meets the policy.
    """
    T = probs.shape[0]
    if strategy == "S1":
        meets = probs.max(axis=-1) > threshold
    elif strategy == "S3":
        meets = _entropy_last(probs) < threshold
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    if min_exit_frame > 0:
        meets = meets.copy()
        meets[:min_exit_frame] = False
    if meets.any():
        return int(np.argmax(meets))
    return T - 1


def decode_frames(frame_bytes: Sequence[bytes]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for b in frame_bytes:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        # Force full decode before the BytesIO object is destroyed.
        frames.append(img.copy())
    return frames


def apply_view(img: Image.Image, view: str, resize: int) -> Image.Image:
    if view == "raw":
        out = img.copy()
    elif view == "student":
        out = img.convert("L").resize((64, 64), Image.BILINEAR).convert("RGB")
    elif view == "teacher":
        out = img.resize((112, 112), Image.BILINEAR).convert("RGB")
    else:
        raise ValueError(f"Unknown view: {view}")

    if resize > 0:
        w, h = out.size
        scale = resize / max(w, h)
        out = out.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.NEAREST if view == "student" else Image.BILINEAR)
    return out


def annotate(img: Image.Image, text: str) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    strip_h = 28
    canvas = Image.new("RGB", (w, h + strip_h), "white")
    canvas.paste(img, (0, strip_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((6, 8), text[:140], fill="black", font=font)
    return canvas


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:120]


def choose_items(items: Sequence[SplitItem], num: int, seed: int, ids: Sequence[str], classes: Sequence[str]) -> list[SplitItem]:
    if ids:
        lookup = {it.video_id: it for it in items}
        missing = [x for x in ids if x not in lookup]
        if missing:
            raise KeyError(f"IDs not found in split CSV: {missing[:10]}")
        return [lookup[x] for x in ids]

    pool = list(items)
    if classes:
        class_set = set(classes)
        pool = [it for it in pool if it.label in class_set]
        if not pool:
            raise ValueError(f"No clips found for classes={classes}")

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:num]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export sample GIFs from Jester HDF5 clips.")
    parser.add_argument("--h5", required=True, type=Path, help="Path to jester_full.h5")
    parser.add_argument("--split_csv", required=True, type=Path, help="Path to jester-v1-train.csv or validation CSV")
    parser.add_argument("--labels_txt", type=Path, default=None, help="Optional jester_labels.txt")
    parser.add_argument("--out", type=Path, default=Path("outputs/jester_gifs"), help="Output directory")
    parser.add_argument("--num", type=int, default=8, help="Number of random samples to export")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--ids", nargs="*", default=[], help="Specific video IDs to export")
    parser.add_argument("--classes", nargs="*", default=[], help="Optional class-name filter, exact match")
    parser.add_argument("--view", choices=["raw", "teacher", "student"], default="student", help="Visualization view")
    parser.add_argument("--frames", choices=["all", "uniform8", "segment8"], default="uniform8",
                        help="all/downsampled raw frames, 8 uniform frames, or 8 TSN-segment-center frames (matches model input)")
    parser.add_argument("--logits_npz", type=Path, default=None,
                        help="Per-frame logits .npz (logits[N,T,K], labels[N]) from early_exit_eval.py. "
                             "If set, annotate each frame with Conf and EXIT marker. "
                             "Implies --frames segment8 unless overridden.")
    parser.add_argument("--strategy", choices=["S1", "S3"], default="S1",
                        help="Early-exit strategy (only used with --logits_npz). "
                             "S1=max-softmax > threshold; S3=entropy < threshold.")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Strategy threshold (default 0.85 = picked C1j operating point).")
    parser.add_argument("--min_exit_frame", type=int, default=5,
                        help="Don't allow exit before this frame index (default 5 = picked C1j floor).")
    parser.add_argument("--uniform_count", type=int, default=8, help="Number of uniformly sampled frames for --frames uniform8")
    parser.add_argument("--max_frames", type=int, default=40, help="Max frames for --frames all")
    parser.add_argument("--fps", type=float, default=8.0, help="GIF frame rate")
    parser.add_argument("--resize", type=int, default=192, help="Resize longest side for GIF display; 0 disables")
    parser.add_argument("--no_text", action="store_true", help="Disable text overlay")
    args = parser.parse_args()

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    labels = read_labels(args.labels_txt)
    items = parse_split_csv(args.split_csv, labels)
    chosen = choose_items(items, args.num, args.seed, args.ids, args.classes)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load precomputed per-frame logits if given. Row i corresponds to
    # jester_common.parse_split("validation")[i], so we build a vid->row map.
    logits_by_vid: dict[str, np.ndarray] | None = None
    if args.logits_npz is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from jester_common import parse_split as jc_parse_split  # noqa: WPS433

        npz = np.load(args.logits_npz)
        logits_arr = npz["logits"]
        val_items = jc_parse_split("validation")
        if len(val_items) != logits_arr.shape[0]:
            raise RuntimeError(
                f"logits row count {logits_arr.shape[0]} != validation split size {len(val_items)}"
            )
        logits_by_vid = {str(vid): logits_arr[i] for i, (vid, _) in enumerate(val_items)}
        # Force segment8 sampling so GIF frames align with logits frames.
        if args.frames == "uniform8":
            print(f"  [info] --logits_npz given; switching frames mode uniform8 -> segment8 "
                  f"to align with TSN sampling used by early_exit_eval")
            args.frames = "segment8"

    manifest_lines = ["video_id,label,gif_path,h5_path,num_raw_frames,num_exported_frames,view,frames_mode"]

    with h5py.File(args.h5, "r") as h5:
        for rank, item in enumerate(chosen):
            obj, h5_path = find_clip_obj(h5, item.video_id)
            raw_bytes = list(iter_frame_bytes_from_obj(obj))
            indices = select_indices(len(raw_bytes), args.frames, args.uniform_count, args.max_frames)
            images = decode_frames([raw_bytes[i] for i in indices])

            gif_frames = []
            label_str = item.label or "unknown"

            # Per-frame confidences + exit-frame (only when logits provided).
            probs_for_sample = None
            exit_frame_idx = None
            if logits_by_vid is not None and item.video_id in logits_by_vid:
                L = logits_by_vid[item.video_id]  # (T_logits, K)
                probs_for_sample = _softmax_last(L)
                exit_frame_idx = decide_exit_frame(
                    probs_for_sample, args.strategy, args.threshold, args.min_exit_frame
                )

            for j, (idx, img) in enumerate(zip(indices, images)):
                vis = apply_view(img, args.view, args.resize)
                if not args.no_text:
                    if probs_for_sample is not None and j < probs_for_sample.shape[0]:
                        conf = float(probs_for_sample[j].max())
                        text = f"Frame {j + 1}/{len(images)}  Conf: {conf:.2f}"
                        if exit_frame_idx is not None and j >= exit_frame_idx:
                            text += "  EXIT"
                    else:
                        text = label_str
                    vis = annotate(vis, text)
                gif_frames.append(vis)

            out_name = f"{rank:02d}_{safe_name(label_str)}_{safe_name(item.video_id)}_{args.view}_{args.frames}.gif"
            out_path = args.out / out_name
            duration_ms = max(1, int(round(1000.0 / args.fps)))
            gif_frames[0].save(
                out_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
            print(f"[OK] {item.video_id} | {label_str} | raw={len(raw_bytes)} frames | exported={len(gif_frames)} -> {out_path}")
            manifest_lines.append(
                f"{item.video_id},{label_str},{out_path},{h5_path},{len(raw_bytes)},{len(gif_frames)},{args.view},{args.frames}"
            )

    manifest = args.out / "manifest.csv"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"\nWrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
