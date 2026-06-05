"""Shared utilities for Jester dataset loading.

Day 2 — single source of truth for:
  * 27-class label list + name→idx map
  * official split CSV parsing (returns aligned (video_id, label_idx) lists)
  * TSN-style segment frame sampling (T frames per clip, train/val variants)
  * horizontal-flip label remap (only for direction-sensitive gesture classes)

All Day 2+ data loaders (PyTorch Pipeline H, TF Pipeline L) consume these
helpers — guarantees both pipelines see identical (video_id, label) pairs in
the same order, which is the prerequisite for resolution-mismatched KD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/data/jester")
H5_PATH = DATA_ROOT / "jester_full.h5"
ANN_DIR = DATA_ROOT / "annotations"
LABELS_TXT = DATA_ROOT / "jester_labels.txt"


def load_class_names() -> list[str]:
    with LABELS_TXT.open() as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_class_index() -> dict[str, int]:
    return {n: i for i, n in enumerate(load_class_names())}


def parse_split(split: str) -> list[tuple[int, int]]:
    """Read official Jester split CSV. Returns list of (video_id, label_idx).

    split in {"train", "validation", "test"}. For "test" the CSV has no labels
    (Jester held-out test set), so we raise — the Plan only uses train/val.
    """
    if split == "test":
        raise ValueError("test split has no labels — not used in Day 2 baseline")
    csv = ANN_DIR / f"jester-v1-{split}.csv"
    name_to_idx = build_class_index()
    out: list[tuple[int, int]] = []
    with csv.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            vid, name = ln.split(";", 1)
            out.append((int(vid), name_to_idx[name]))
    return out


# ---- Frame sampling (TSN-style segment sampling, 0-indexed) -----------------

def sample_segment_indices(num_frames: int, T: int, training: bool) -> np.ndarray:
    """TSN-style sampling: divide clip into T equal segments, pick one frame each.

    training=True : random offset within segment (temporal jitter)
    training=False: center of each segment (deterministic for val/test)

    Returns shape (T,) int64 array, values in [0, num_frames).
    Mirrors MIT TSM ops/dataset.py _sample_indices / _get_val_indices but
    0-indexed (HDF5 is 0-indexed; TSM repo is 1-indexed for file naming).
    """
    if num_frames >= T:
        avg = num_frames / float(T)
        if training:
            base = (np.arange(T) * avg).astype(np.int64)
            jitter = np.random.randint(0, max(1, int(avg)), size=T)
            offsets = np.minimum(base + jitter, num_frames - 1)
        else:
            offsets = (np.arange(T) * avg + avg / 2.0).astype(np.int64)
            offsets = np.minimum(offsets, num_frames - 1)
    else:
        offsets = np.minimum(np.arange(T), num_frames - 1).astype(np.int64)
    return offsets


# ---- Horizontal flip label remap --------------------------------------------
# Only the four gesture pairs whose semantics genuinely invert under mirror.
# For all other classes the label is preserved (= identity in the map).
#
# Why this matters: blind hflip pollutes the label space — e.g. "Swiping Left"
# flipped looks exactly like "Swiping Right". Either remap the label or drop
# the augmentation for those classes. We remap so we don't waste samples.

_FLIP_PAIRS_NAMES = [
    ("Swiping Left", "Swiping Right"),
    ("Sliding Two Fingers Left", "Sliding Two Fingers Right"),
    ("Turning Hand Clockwise", "Turning Hand Counterclockwise"),
    # Rolling Hand Forward/Backward is a pitch-axis (horizontal) rotation;
    # horizontal mirror does NOT swap chirality. Leave them out.
]


def build_flip_label_map() -> np.ndarray:
    """27-long lookup. flip_map[label] = flipped label."""
    name_to_idx = build_class_index()
    n_class = len(name_to_idx)
    m = np.arange(n_class, dtype=np.int64)
    for a, b in _FLIP_PAIRS_NAMES:
        ia, ib = name_to_idx[a], name_to_idx[b]
        m[ia], m[ib] = ib, ia
    return m


FLIP_LABEL_MAP = build_flip_label_map()


__all__ = [
    "DATA_ROOT", "H5_PATH", "ANN_DIR", "LABELS_TXT",
    "load_class_names", "build_class_index", "parse_split",
    "sample_segment_indices", "FLIP_LABEL_MAP", "build_flip_label_map",
]
