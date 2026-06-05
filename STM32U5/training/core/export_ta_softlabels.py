"""Day 7 deliverable: export TA (TSM-MobileNetV2) per-frame soft labels.

Loads `models/teacher_assistant/best.pt`, forwards over the train split with
the canonical `parse_split('train')` order, and stores per-frame pre-softmax
logits as float16 — shape `[N_train, T=8, 27]`, indexed by position in
`parse_split('train')`.

This is the **Stage-2 KD input for Day 8** (student variants V0-V6 training).
The student sees 64×64 grayscale, but learns the high-resolution TA's
per-frame logit distribution (resolution-mismatched KD, default per Plan).

Single-GPU is enough — 118562 clips × T=8 × 27 classes is ~52 MB output.
With `T=8` forward per clip in batches of 64, ~1850 forward steps ≈ 8-15 min
wall on A100.

Per Plan Section 1 Note: Swin-T exports video-level; TA exports per-frame
(utilizes the causal uni-directional shift — every frame's logits are
independently meaningful, t=7 is the most informed).

Optional --tta flag: average orig + hflipped views with FLIP_LABEL_MAP applied
to the flipped-view class axis (same convention as Day 5 Swin-T TTA).
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
from ta_tsm_mbv2 import TaTSMMobileNetV2


PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=PROJECT_ROOT / "models/teacher_assistant/best.pt")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "soft_labels/soft_labels_ta_train.npy")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--n_div", type=int, default=8)
    parser.add_argument("--tta", action="store_true",
                        help="average orig + hflipped (with FLIP_LABEL_MAP class permutation)")
    parser.add_argument("--eval_val", action="store_true",
                        help="also run a val pass and print top-1 (sanity vs train run's epoch log)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load TA model (no ImageNet redownload — load_state_dict overwrites it)
    model = TaTSMMobileNetV2(
        n_classes=27, n_segment=args.T, n_div=args.n_div,
        dropout=0.5, pretrained=False,
    ).to(device)
    payload = torch.load(args.ckpt, map_location=device)
    state_dict = payload["model"] if "model" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[export] loaded {args.ckpt} (epoch={payload.get('epoch','?')}, "
          f"best_val_acc={payload.get('best_val_acc','?')})", flush=True)

    flip_map = torch.from_numpy(FLIP_LABEL_MAP).to(device)

    # ---- Optional val sanity check ----
    if args.eval_val:
        val_ds = JesterH5Torch(split="validation", T=args.T, img_size=args.img_size,
                               training=False, flip_label_aware=True)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
            persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init,
        )
        n_correct = 0
        n_total = 0
        n_correct_tta = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)               # (B, T, 27)
                clip_logits = logits[:, -1]         # last-frame prediction
                n_correct += (clip_logits.argmax(dim=1) == y).sum().item()
                if args.tta:
                    x_f = torch.flip(x, dims=(-1,))
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits_f = model(x_f)
                    logits_f_unflipped = logits_f[:, :, flip_map]
                    avg = (logits + logits_f_unflipped) / 2.0
                    n_correct_tta += (avg[:, -1].argmax(dim=1) == y).sum().item()
                n_total += y.size(0)
        print(f"[val] top1 = {n_correct / n_total:.4f}  (last-frame)", flush=True)
        if args.tta:
            print(f"[val] top1 TTA = {n_correct_tta / n_total:.4f}  (last-frame, hflip avg)", flush=True)

    # ---- Train per-frame logits export ----
    # IMPORTANT: training=False + flip_label_aware=False (no aug, deterministic
    # frame sampling, no hflip) — keeps DataLoader iteration order = parse_split()
    items = parse_split(args.split)
    N = len(items)
    ds = JesterH5Torch(
        split=args.split, T=args.T, img_size=args.img_size,
        training=False, flip_label_aware=False,
    )
    assert len(ds) == N, f"dataset len mismatch: {len(ds)} vs parse_split {N}"
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init,
    )

    out = np.zeros((N, args.T, 27), dtype=np.float16)
    t0 = time.time()
    cursor = 0
    if args.tta:
        print("[export] TTA enabled: averaging orig + hflipped per-frame logits", flush=True)

    with torch.no_grad():
        for step, (x, _y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)                          # (B, T, 27)
            if args.tta:
                x_f = torch.flip(x, dims=(-1,))
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits_f = model(x_f)                  # (B, T, 27) — flipped-view class order
                # Permute the class axis through FLIP_LABEL_MAP so directional
                # classes (Swiping Left ↔ Right etc.) align with the original
                # label space, then average. T-axis untouched (temporal order
                # is preserved under hflip).
                logits_f_unflipped = logits_f[:, :, flip_map]
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
    el = time.time() - t0
    print(f"[export] wrote {args.out}  shape={out.shape}  dtype={out.dtype}  "
          f"size={args.out.stat().st_size/1e6:.2f}MB  total={el/60:.1f}m",
          flush=True)

    # Health check: per-frame argmax distribution + entropy across the full file
    pred_per_frame = out.argmax(axis=-1)                  # (N, T) int
    train_acc_per_frame = (pred_per_frame == np.asarray([y for _, y in items])[:, None]).mean(axis=0)
    print("[health] train top-1 per frame:", [f"{a:.4f}" for a in train_acc_per_frame.tolist()])
    # Mean confidence at t=7
    p = np.exp(out[:, -1] - out[:, -1].max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    mean_top1_prob = p.max(axis=1).mean()
    mean_entropy = -(p * np.log(p + 1e-12)).sum(axis=1).mean()
    print(f"[health] t=7 mean top-1 prob = {mean_top1_prob:.4f}, mean entropy = {mean_entropy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
