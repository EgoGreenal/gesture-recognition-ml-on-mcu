"""Day 2 baseline TSN: 2D ResNet-18 on Jester, 1 epoch, no TSM, no KD.

Purpose: sanity-check the entire training infrastructure (Pipeline H loader,
data augmentation, mixed-precision training, val pipeline) end-to-end before
investing GPU hours in Stage 1/2 (Swin-T, TSM-MBV2). Accept if val top-1 > 55%.

Model:
  ResNet-18 (ImageNet pretrained) with FC -> Linear(512, 27).
  Forward: (B, T, 3, 112, 112) -> reshape (B*T, 3, 112, 112) -> backbone
           -> (B*T, 27) -> reshape (B, T, 27) -> mean over T -> (B, 27)
  This is the standard TSN consensus head (no TSM = no temporal info beyond
  late fusion; expected to underperform full TSM but should easily clear 55%).

Hyperparameters (deliberate baseline choices, not tuned):
  batch=64, T=8, lr=1e-3, optimizer=Adam, 1 epoch.
  Mixed precision (autocast + GradScaler) to fit comfortably on A100.

Outputs:
  models/baseline_tsn.pth          — final state_dict
  logs/baseline_tsn_<jobid>.json   — accuracy + timing metadata
  stdout — per-100-step train log + val summary
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torchvision.models import ResNet18_Weights, resnet18

from data_loader_torch import build_dataloader


PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
N_CLASSES = 27


# ---- Model -----------------------------------------------------------------

class TSNResNet18(nn.Module):
    """2D ResNet-18 backbone + per-frame FC + temporal mean (TSN consensus)."""

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_feat = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        logits = self.backbone(x)             # (B*T, n_class)
        logits = logits.view(b, t, -1).mean(dim=1)  # (B, n_class)  TSN consensus
        return logits


# ---- Train / Eval loops ----------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device, use_amp: bool) -> tuple[float, float]:
    model.eval()
    n = 0
    top1 = 0
    top5 = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
        _, top5_pred = logits.topk(5, dim=1)
        correct1 = (top5_pred[:, 0] == y).sum().item()
        correct5 = (top5_pred == y.unsqueeze(1)).any(dim=1).sum().item()
        n += y.size(0)
        top1 += correct1
        top5 += correct5
    return top1 / n, top5 / n


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler: GradScaler,
    device,
    use_amp: bool,
    log_every: int = 100,
) -> tuple[float, float]:
    model.train()
    crit = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_correct = 0
    running_n = 0
    t_start = time.time()

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = crit(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            running_correct += (pred == y).sum().item()
            running_n += y.size(0)
            running_loss += loss.item() * y.size(0)

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_start
            ips = running_n / max(elapsed, 1e-6)
            avg_loss = running_loss / running_n
            acc = running_correct / running_n
            print(
                f"  step {step+1:5d}  "
                f"loss={avg_loss:.4f}  acc={acc:.4f}  "
                f"ips={ips:.1f}  elapsed={elapsed/60:.1f}min",
                flush=True,
            )

    final_loss = running_loss / max(running_n, 1)
    final_acc = running_correct / max(running_n, 1)
    return final_loss, final_acc


# ---- Main ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--out_dir", type=Path, default=PROJECT_ROOT / "models"
    )
    parser.add_argument(
        "--log_dir", type=Path, default=PROJECT_ROOT / "logs"
    )
    parser.add_argument("--job_id", type=str, default=os.environ.get("SLURM_JOB_ID", "local"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("FATAL: CUDA unavailable. Baseline must run on GPU.", flush=True)
        return 1
    device = torch.device("cuda")
    use_amp = not args.no_amp

    print(f"[baseline_tsn] device={device}  amp={use_amp}", flush=True)
    print(f"[baseline_tsn] args={vars(args)}", flush=True)

    # Loaders
    t0 = time.time()
    train_loader = build_dataloader(
        "train",
        batch_size=args.batch_size,
        T=args.T,
        img_size=args.img_size,
        training=True,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        "validation",
        batch_size=args.batch_size,
        T=args.T,
        img_size=args.img_size,
        training=False,
        num_workers=args.num_workers,
    )
    print(
        f"[baseline_tsn] train: {len(train_loader.dataset)} clips, "
        f"{len(train_loader)} steps/epoch",
        flush=True,
    )
    print(f"[baseline_tsn] val:   {len(val_loader.dataset)} clips", flush=True)

    # Model
    model = TSNResNet18().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[baseline_tsn] model params: {n_params/1e6:.2f}M", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=use_amp)

    # Train
    train_t0 = time.time()
    for ep in range(args.epochs):
        print(f"\n=== Epoch {ep+1}/{args.epochs} ===", flush=True)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp
        )
        print(
            f"[epoch {ep+1}] train_loss={train_loss:.4f}  train_acc={train_acc:.4f}",
            flush=True,
        )
    train_time = time.time() - train_t0

    # Eval
    print("\n=== Validation ===", flush=True)
    val_t0 = time.time()
    val_top1, val_top5 = evaluate(model, val_loader, device, use_amp)
    val_time = time.time() - val_t0
    print(
        f"[val] top1={val_top1:.4f}  top5={val_top5:.4f}  ({val_time:.1f}s)",
        flush=True,
    )

    # Persist
    ckpt_path = args.out_dir / "baseline_tsn.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[baseline_tsn] saved checkpoint -> {ckpt_path}", flush=True)

    # vars(args) returns a reference to argparse's internal dict — copy + cast
    # Path values to str so JSON serializes (and we don't mutate args itself).
    args_serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    result = {
        "job_id": args.job_id,
        "args": args_serializable,
        "n_params_M": n_params / 1e6,
        "train_loss_final": train_loss,
        "train_acc_final": train_acc,
        "val_top1": val_top1,
        "val_top5": val_top5,
        "train_time_s": train_time,
        "val_time_s": val_time,
        "total_time_s": time.time() - t0,
        "accepted_55": val_top1 > 0.55,
    }
    log_path = args.log_dir / f"baseline_tsn_{args.job_id}.json"
    with log_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[baseline_tsn] wrote metrics -> {log_path}", flush=True)

    verdict = "PASS (>55%)" if result["accepted_55"] else "BELOW THRESHOLD (≤55%)"
    print(f"\n[baseline_tsn] VERDICT: {verdict}  (val_top1={val_top1:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
