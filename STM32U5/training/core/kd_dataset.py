"""Day 6 helper: wrap JesterH5Torch + Swin-T soft-labels npy for KD training.

Returns (clip, label, soft_logits) per item. `soft_logits` shape: (27,) FP32.

Flip-aware soft-label remap (CRITICAL):
  When the underlying loader applies a horizontal flip during augmentation,
  it also remaps the hard label via FLIP_LABEL_MAP (e.g. Swiping Left -> Right).
  The Swin-T soft logits were exported on the **un-flipped** view, so they
  encode "video looks like Swiping Left" — class probabilities live in the
  un-flipped label space. To match the flipped clip, we permute the soft
  vector with FLIP_LABEL_MAP: `soft_flipped = soft[flip_map]`. This is the
  symmetric operation that `export_softlabels.py` does in TTA mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loader_torch import JesterH5Torch
from jester_common import FLIP_LABEL_MAP, parse_split


class JesterH5KD(Dataset):
    """Wraps JesterH5Torch (return_flip=True) + a video-level soft-labels npy.

    Args:
        split: "train" / "validation". Soft labels are only meaningful for
            "train" (Day 5 only exported train).
        soft_labels_path: path to .npy file with shape (N, 27) float16,
            indexed in `parse_split(split)` order.
        T, img_size: passed through to JesterH5Torch.
        training: passed through; flips only happen when True.
    """

    def __init__(
        self,
        split: str,
        soft_labels_path: Path | str,
        T: int = 8,
        img_size: int = 112,
        training: bool = True,
    ) -> None:
        super().__init__()
        self.base = JesterH5Torch(
            split=split,
            T=T,
            img_size=img_size,
            training=training,
            flip_label_aware=True,
            return_flip=True,
        )
        self.split = split
        # Load full npy in memory: 118562 * 27 * 2 bytes = 6.4 MB, trivial.
        self.soft = np.load(str(soft_labels_path))  # (N, 27) float16
        n_expected = len(parse_split(split))
        if self.soft.shape != (n_expected, 27):
            raise RuntimeError(
                f"soft labels shape {self.soft.shape} != expected ({n_expected}, 27)"
            )
        self.flip_map = np.asarray(FLIP_LABEL_MAP, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        clip, label, do_flip = self.base[idx]
        soft = self.soft[idx]  # (27,) float16
        if do_flip:
            soft = soft[self.flip_map]
        # Cast to FP32 here so downstream is dtype-stable (loss casts to FP32
        # anyway); also ensures pin_memory + non_blocking transfer is safe.
        soft_t = torch.from_numpy(soft.astype(np.float32, copy=False))
        return clip, label, soft_t


if __name__ == "__main__":
    PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
    soft = PROJECT_ROOT / "soft_labels" / "soft_labels_train_swa_tta.npy"
    ds = JesterH5KD(split="train", soft_labels_path=soft, training=True)
    print(f"len(ds) = {len(ds)}")
    c, y, s = ds[0]
    print(f"  clip: {tuple(c.shape)} dtype={c.dtype}")
    print(f"  label: {y}")
    print(f"  soft: {tuple(s.shape)} dtype={s.dtype}  argmax={s.argmax().item()}  max={s.max().item():.3f}")
    # Run a few more to exercise the flip path
    n_flipped = 0
    for i in range(20):
        c, y, s = ds[i]
    print(f"sampled 20 items OK")
