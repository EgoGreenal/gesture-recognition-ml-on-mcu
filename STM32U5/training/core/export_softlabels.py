"""Day 5 deliverable: export Super Teacher video-level soft labels to .npy.

Runs after training. Loads `best.pt` (or specified ckpt), forwards over the
train split with the canonical parse_split order, and stores the pre-softmax
logits as float16 — shape [N_train, 27], indexed by position in parse_split('train').

This is Stage-1 KD input for Day 6 (TA training). Indices align with
parse_split('train') in jester_common.py; the TA trainer reads the same
parse_split to index into this npy.

Single-GPU is enough — 118562 clips * batch 64 ≈ 1850 forward steps,
~5-10 min wall on A100. No DDP needed.

Per Plan Section 1 Note: Swin-T is bidirectional → only video-level
predictions are exported (not per-frame).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader_torch import JesterH5Torch, _worker_init
from jester_common import FLIP_LABEL_MAP, parse_split
from swin_t_video import SuperTeacherSwinT


PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=PROJECT_ROOT / "models/super_teacher/best.pt")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "soft_labels/soft_labels_train.npy")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--tta", action="store_true",
                        help="average logits of orig + hflipped (with FLIP_LABEL_MAP applied)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model = SuperTeacherSwinT(n_classes=27, pretrained_2d=None).to(device)
    payload = torch.load(args.ckpt, map_location=device)
    state_dict = payload["model"] if "model" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[export] loaded {args.ckpt} (epoch={payload.get('epoch','?')}, "
          f"best_val_acc={payload.get('best_val_acc','?')})", flush=True)

    # Deterministic loader (training=False → center crop + segment-center frames)
    # IMPORTANT: shuffle=False keeps DataLoader iteration order = parse_split() order
    items = parse_split(args.split)
    N = len(items)
    ds = JesterH5Torch(split=args.split, T=args.T, img_size=args.img_size, training=False)
    assert len(ds) == N, f"dataset len mismatch: {len(ds)} vs parse_split {N}"
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init,
    )

    # Forward (with optional TTA — average logits of orig + hflipped)
    flip_map = torch.from_numpy(FLIP_LABEL_MAP).to(device)
    out = np.zeros((N, 27), dtype=np.float16)
    t0 = time.time()
    cursor = 0
    if args.tta:
        print("[export] TTA enabled: averaging logits of orig + hflipped views", flush=True)
    with torch.no_grad():
        for step, (x, _y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
            if args.tta:
                x_f = torch.flip(x, dims=(-1,))
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits_f = model(x_f)
                # Permute output classes through FLIP_LABEL_MAP so the flipped
                # view's logits are in the same label space as the original.
                logits_f_unflipped = logits_f[:, flip_map]
                logits = (logits + logits_f_unflipped) / 2.0
            n = logits.size(0)
            out[cursor:cursor + n] = logits.float().cpu().numpy().astype(np.float16)
            cursor += n
            if (step + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  step {step+1}/{len(loader)}  "
                      f"clips {cursor}/{N}  ips={cursor/el:.1f}  el={el/60:.1f}m",
                      flush=True)
    assert cursor == N
    np.save(args.out, out)
    print(f"[export] wrote {args.out}  shape={out.shape}  dtype={out.dtype}  "
          f"size={args.out.stat().st_size/1e6:.2f}MB  total={(time.time()-t0)/60:.1f}m",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
