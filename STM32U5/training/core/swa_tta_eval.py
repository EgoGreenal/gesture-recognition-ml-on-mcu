"""Day 5 path-B exploration: SWA (Stochastic Weight Averaging) + TTA (Test-Time Aug).

Two parallel boosts to try to push Super Teacher val_top1 from 94.71% to ≥95% (Plan
section 4 Day 5 acceptance band) without further training.

SWA:
  Average the model state_dicts of epoch_25, 26, 27, 28, 29, 30 (6 plateau-region
  checkpoints from the cosine-decay tail). Saves `models/super_teacher/swa.pt`.
  Doesn't touch best.pt; SWA is a parallel candidate.

TTA:
  For each val clip, forward both the original and the horizontally-flipped version.
  Sum probabilities (in label space) and argmax. Because Jester has 3 direction-
  sensitive class pairs (Swiping L/R, Sliding 2F L/R, Turning CW/CCW), the flipped
  forward's logits must be permuted through FLIP_LABEL_MAP before averaging — else
  flipping pollutes the directional classes. This is mirror-symmetric of the
  flip-aware augmentation in data_loader_torch.

Outputs:
  models/super_teacher/swa.pt      — averaged weights (no optimizer/scaler state)
  logs/swa_tta_eval.json           — top-1 / top-5 for each of:
                                       (best.pt + no_tta), (best.pt + tta),
                                       (swa.pt + no_tta),  (swa.pt + tta)

Single GPU, ~10 min wall.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader_torch import JesterH5Torch, _worker_init
from jester_common import FLIP_LABEL_MAP
from swin_t_video import SuperTeacherSwinT


PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")


def average_checkpoints(ckpt_paths: list[Path]) -> dict:
    """Per-key mean of model state_dicts. Buffers (running stats) also averaged.

    BN-related running_mean/running_var: averaging is approximate; the precise
    SWA recipe re-runs train data to recompute BN stats. Swin-T uses LayerNorm
    (no running stats) — so the approximation is exact for our model. Linear/
    Conv weights/biases average cleanly. relative_position_bias_table averages
    cleanly too (it's a learnable tensor, no buffer ambiguity).
    """
    print(f"[swa] averaging {len(ckpt_paths)} checkpoints:")
    summed: dict[str, torch.Tensor] | None = None
    for i, p in enumerate(ckpt_paths):
        print(f"  [{i+1}/{len(ckpt_paths)}] loading {p.name}")
        payload = torch.load(p, map_location="cpu")
        sd = payload["model"] if "model" in payload else payload
        if summed is None:
            summed = {k: v.detach().clone().float() for k, v in sd.items()}
        else:
            for k in summed:
                summed[k] += sd[k].detach().float()
    assert summed is not None
    averaged = {k: (v / len(ckpt_paths)).to(torch.float32) for k, v in summed.items()}
    return averaged


@torch.no_grad()
def eval_with_optional_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool,
    flip_map: torch.Tensor,
) -> tuple[float, float, int]:
    """Return (top1, top5, n_samples).

    tta=False: standard center-crop forward
    tta=True:  average probabilities of original + horizontally-flipped input,
               with FLIP_LABEL_MAP applied to the flipped output before averaging
    """
    model.eval()
    n_correct1 = 0
    n_correct5 = 0
    n_total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits_o = model(x)
        probs = torch.softmax(logits_o, dim=1)

        if tta:
            # x shape: (B, T, 3, H, W) — flip W dim (last)
            x_f = torch.flip(x, dims=(-1,))
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits_f = model(x_f)
            # Model predicts on flipped image; permute output labels back to
            # original label space: if class c is what the flipped image looks
            # like, then in the original frame it was FLIP_LABEL_MAP[c].
            # Equivalently we permute the logits columns by the inverse map
            # (which is the same since FLIP_LABEL_MAP is an involution).
            logits_f_unflipped = logits_f[:, flip_map]
            probs_f = torch.softmax(logits_f_unflipped, dim=1)
            probs = (probs + probs_f) / 2.0

        _, top5_pred = probs.topk(5, dim=1)
        n_correct1 += (top5_pred[:, 0] == y).sum().item()
        n_correct5 += (top5_pred == y.unsqueeze(1)).any(dim=1).sum().item()
        n_total += y.size(0)
    return n_correct1 / n_total, n_correct5 / n_total, n_total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", type=Path,
        default=PROJECT_ROOT / "models/super_teacher",
    )
    parser.add_argument("--epochs_for_swa", type=int, nargs="+",
                        default=[25, 26, 27, 28, 29, 30])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--log_path", type=Path,
                        default=PROJECT_ROOT / "logs" / "swa_tta_eval.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    # Need plateau snapshots; we only saved every 5 epochs (5/10/15/20/25/30),
    # so 25-30 mid-epoch ones don't exist. Check what's available.
    avail_paths = []
    for ep in args.epochs_for_swa:
        p = args.ckpt_dir / f"epoch_{ep:02d}.pt"
        if p.is_file():
            avail_paths.append(p)
        else:
            print(f"[swa] warning: {p.name} not found, skipping")
    print(f"[swa] using {len(avail_paths)} checkpoints: {[p.name for p in avail_paths]}")
    if len(avail_paths) < 2:
        raise RuntimeError("need >=2 snapshots for averaging")

    # ---- Build SWA checkpoint ----
    swa_sd = average_checkpoints(avail_paths)
    swa_path = args.ckpt_dir / "swa.pt"
    torch.save({"model": swa_sd, "epoch": "swa", "best_val_acc": None}, swa_path)
    print(f"[swa] saved {swa_path} ({swa_path.stat().st_size/1e6:.1f}MB)")

    # ---- Val loader (deterministic) ----
    val_ds = JesterH5Torch(split="validation", T=args.T, img_size=args.img_size, training=False)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init,
    )
    flip_map = torch.from_numpy(FLIP_LABEL_MAP).to(device)
    print(f"[eval] val: {len(val_ds)} clips, {len(val_loader)} batches")

    # ---- Evaluate combinations ----
    # Include best.pt only if it exists in the ckpt_dir (Day 5 path-B re-runs
    # eval on super_teacher_ft/, where there's no best.pt — skip silently).
    results = {}
    ckpt_list = []
    best_path = args.ckpt_dir / "best.pt"
    if best_path.is_file():
        ckpt_list.append(("best", best_path))
    # Also allow evaluating the ORIGINAL super_teacher/best.pt as anchor
    orig_best = PROJECT_ROOT / "models/super_teacher/best.pt"
    if orig_best.is_file() and orig_best != best_path:
        ckpt_list.append(("orig_best", orig_best))
    # Per-snapshot eval (helpful to know which fine-tune epoch is strongest)
    for ep in args.epochs_for_swa:
        p = args.ckpt_dir / f"epoch_{ep:02d}.pt"
        if p.is_file():
            ckpt_list.append((f"ep{ep}", p))
    ckpt_list.append(("swa", swa_path))
    for ckpt_name, ckpt_path in ckpt_list:
        model = SuperTeacherSwinT(n_classes=27, pretrained_2d=None).to(device)
        payload = torch.load(ckpt_path, map_location=device)
        sd = payload["model"] if "model" in payload else payload
        # SWA sd was saved fp32, but model may be fp32 too — load_state_dict handles
        model.load_state_dict(sd)
        for tta_flag in (False, True):
            t0 = time.time()
            top1, top5, n = eval_with_optional_tta(model, val_loader, device, tta_flag, flip_map)
            dt = time.time() - t0
            key = f"{ckpt_name}_{'tta' if tta_flag else 'no_tta'}"
            results[key] = {"top1": top1, "top5": top5, "n": n, "wall_s": dt}
            print(f"[{key:14s}] top1={top1:.4f}  top5={top5:.4f}  n={n}  wall={dt:.1f}s")

    # ---- Persist + summarize ----
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    with args.log_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[swa_tta] wrote {args.log_path}")

    base = results["best_no_tta"]["top1"]
    print(f"\nSummary (Δ vs best+no_tta = {base:.4f}):")
    for k, v in results.items():
        delta = v["top1"] - base
        flag = " ★" if v["top1"] >= 0.95 else ""
        print(f"  {k:14s}  top1={v['top1']:.4f}  Δ={delta:+.4f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
