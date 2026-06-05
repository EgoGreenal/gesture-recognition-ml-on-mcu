"""PyTorch DataLoader — Pipeline H (112x112 RGB) for Teacher / TA.

Backed by data/jester/jester_full.h5 (vlen uint8 JPEG bytes per clip frame).

Design notes:
  * h5py.File handle opened LAZILY per worker (Python pickles the Dataset
    when spawning workers — a live h5 handle is non-picklable, and even if
    it were, sharing across forked workers on Lustre corrupts reads).
  * Frame sampling: TSN-style T-segment (default T=8) via jester_common.
  * Augmentation (training only): RandomResizedCrop, RandomHorizontalFlip
    with label remap (FLIP_LABEL_MAP), then ImageNet mean/std normalize.
  * Val/test: deterministic center-segment sampling + center crop.
  * Resolution: 112x112 RGB (Plan Section 1: Teacher/TA training resolution).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import io

from jester_common import (
    H5_PATH,
    parse_split,
    sample_segment_indices,
    FLIP_LABEL_MAP,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class JesterH5Torch(Dataset):
    """Pipeline H: HDF5-backed Jester dataset, 112x112 RGB, T frames per clip.

    Args:
        split: "train" or "validation".
        T: frames per clip (8 in this project).
        img_size: spatial side (default 112).
        training: True -> augmented + random sampling; False -> deterministic.
        h5_path: override HDF5 path (default: data/jester/jester_full.h5).
        flip_label_aware: if True, hflip remaps directional labels via
            FLIP_LABEL_MAP. If False, hflip is disabled (label-safe).
    """

    def __init__(
        self,
        split: str,
        T: int = 8,
        img_size: int = 112,
        training: bool = True,
        h5_path: Path | str = H5_PATH,
        flip_label_aware: bool = True,
        return_flip: bool = False,
    ) -> None:
        super().__init__()
        self.split = split
        self.T = T
        self.img_size = img_size
        self.training = training
        self.h5_path = str(h5_path)
        self.flip_label_aware = flip_label_aware
        # If True, __getitem__ returns (clip, label, do_flip) so callers (e.g.
        # the KD wrapper) can mirror the soft-label vector through FLIP_LABEL_MAP.
        self.return_flip = return_flip

        self.items: list[tuple[int, int]] = parse_split(split)
        self._h5: h5py.File | None = None

        # Pre-built transform pipelines (per-frame). Crop applied at PIL stage
        # to share parameters across frames; normalize after stacking.
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    # h5 handle is opened on first access in each worker (avoids fork issues)
    def _h5_ref(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
        return self._h5

    def __len__(self) -> int:
        return len(self.items)

    def _decode_frame(self, raw: bytes) -> Image.Image:
        # Jester source frames are 176x100 JPEG; decode to RGB
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _sample_crop_params(self, w: int, h: int) -> tuple[int, int, int, int]:
        """RandomResizedCrop-style params shared across all T frames.

        Returns (top, left, ch, cw). For training: scale in [0.6, 1.0], aspect
        in [0.9, 1.1] (tight: Jester gestures fill most of the frame, don't
        want to crop hand out). For eval: center crop at full short-side.
        """
        if self.training:
            for _ in range(10):
                target_area = np.random.uniform(0.6, 1.0) * w * h
                aspect = np.random.uniform(0.9, 1.1)
                cw = int(round(np.sqrt(target_area * aspect)))
                ch = int(round(np.sqrt(target_area / aspect)))
                if 0 < cw <= w and 0 < ch <= h:
                    top = np.random.randint(0, h - ch + 1)
                    left = np.random.randint(0, w - cw + 1)
                    return top, left, ch, cw
            # fallback: center crop
        side = min(w, h)
        top = (h - side) // 2
        left = (w - side) // 2
        return top, left, side, side

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        h5 = self._h5_ref()
        vid, label = self.items[idx]
        ds = h5["clips"][str(vid)]
        n_frames = ds.shape[0]

        offsets = sample_segment_indices(n_frames, self.T, self.training)

        # decode first frame to get size (Jester clips share frame size)
        first = self._decode_frame(bytes(ds[int(offsets[0])]))
        w, h = first.size
        top, left, ch, cw = self._sample_crop_params(w, h)
        do_flip = self.training and (np.random.rand() < 0.5)

        frames = []
        for k, off in enumerate(offsets.tolist()):
            img = first if k == 0 else self._decode_frame(bytes(ds[int(off)]))
            img = F.crop(img, top, left, ch, cw)
            img = F.resize(img, [self.img_size, self.img_size])
            if do_flip:
                img = F.hflip(img)
            arr = np.asarray(img, dtype=np.uint8)  # H,W,3
            frames.append(arr)

        clip = np.stack(frames, axis=0)  # T,H,W,3 uint8
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float().div_(255.0)  # T,3,H,W
        clip = (clip - self._mean) / self._std

        if do_flip and self.flip_label_aware:
            label = int(FLIP_LABEL_MAP[label])
        elif do_flip and not self.flip_label_aware:
            # label_aware=False should be paired with no flip; guard
            raise RuntimeError("hflip occurred with flip_label_aware=False")

        if self.return_flip:
            return clip, label, bool(do_flip)
        return clip, label


def _worker_init(_worker_id: int) -> None:
    # Re-seed NumPy from torch initial seed (PyTorch only re-seeds Python RNG).
    info = torch.utils.data.get_worker_info()
    if info is not None:
        seed = (info.seed) % (2**32)
        np.random.seed(seed)


def build_dataloader(
    split: str,
    batch_size: int,
    T: int = 8,
    img_size: int = 112,
    training: bool = True,
    num_workers: int = 4,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = training
    if drop_last is None:
        drop_last = training
    ds = JesterH5Torch(split=split, T=T, img_size=img_size, training=training)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init,
    )


if __name__ == "__main__":
    # Quick sanity: load 2 batches, print shapes + label histogram
    import time

    dl = build_dataloader("validation", batch_size=4, num_workers=0, training=False)
    t0 = time.time()
    for i, (x, y) in enumerate(dl):
        print(f"batch {i}: x={tuple(x.shape)} dtype={x.dtype}  y={y.tolist()}")
        print(f"  mean={x.mean().item():+.3f}  std={x.std().item():.3f}")
        if i >= 1:
            break
    print(f"loaded 2 batches in {time.time()-t0:.2f}s")
