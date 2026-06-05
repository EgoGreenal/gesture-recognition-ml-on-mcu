"""Day 6: Train Teacher Assistant (TSM-MobileNetV2, uni-directional) with KD.

Launch with torchrun:
  torchrun --standalone --nproc_per_node=2 train_ta.py [args]

Pipeline H (112×112 RGB, 8 frames). KD from Swin-T video-level soft labels.

Plan (Section 4 Day 6):
  * 1 node × 2 GPU partial-node + DDP (Day 5 learning: 2-GPU mixed-state slot
    is ~30min queue vs 4-GPU full-node ~10h)
  * per-GPU batch 64, effective batch 128
  * lr 5e-4 × world_size=2 = 1e-3 (linear scaling)
  * SyncBatchNorm (MobileNetV2 has BN in every block — critical for stable
    cross-GPU stats at this effective batch)
  * 15 epoch, AdamW, 2-epoch warmup + cosine to 0
  * AMP fp16
  * Per-epoch ckpt save (Day 5 learned: every-5 too coarse for SWA/best
    selection on a 15-epoch run)
  * Best.pt + latest.pt + epoch_XX.pt

KD loss (kd_loss.py):
  alpha = 0.5, temperature = 4, frame_weights = [0,0,0,0,.05,.1,.2,.65]

Acceptance (per Plan Day 7): val_top1 (taking last-frame logits as the clip's
prediction) >= 91%. < 88% triggers Alignment-TA fallback (Day 7/11).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_loader_torch import JesterH5Torch, _worker_init
from kd_dataset import JesterH5KD
from kd_loss import PerFrameKDLoss, DEFAULT_FRAME_WEIGHTS
from ta_tsm_mbv2 import TaTSMMobileNetV2
# Reuse DDP/optimizer/schedule helpers from the Swin-T trainer
from train_super_teacher import (
    cleanup_ddp,
    is_main,
    log,
    lr_at_step,
    maybe_resume,
    save_ckpt,
    set_lr,
    setup_ddp,
)


PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
DEFAULT_SOFT_LABELS = PROJECT_ROOT / "soft_labels/soft_labels_train_swa_tta.npy"
FALLBACK_SOFT_LABELS = PROJECT_ROOT / "soft_labels/soft_labels_train.npy"


# ---- Optimizer (AdamW, no-decay on BN/bias) --------------------------------

def build_optimizer_ta(model: nn.Module, lr: float, weight_decay: float):
    """AdamW: BN params + biases get no weight decay (standard MobileNetV2 recipe)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


# ---- Distributed eval -----------------------------------------------------

@torch.no_grad()
def evaluate_ta_ddp(
    model: DDP, loader: DataLoader, device: torch.device,
    last_frame_only: bool = True,
) -> tuple[float, float]:
    """val top1/top5. Uses last-frame logits as the clip prediction (matches
    inference + the per-frame weighting in the loss, which puts 65% weight on
    t=7). If last_frame_only=False, averages logits across all frames.
    """
    model.eval()
    local_top1 = torch.zeros(1, device=device)
    local_top5 = torch.zeros(1, device=device)
    local_n = torch.zeros(1, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)            # (B, T, K)
        if last_frame_only:
            clip_logits = logits[:, -1]  # (B, K)
        else:
            clip_logits = logits.mean(dim=1)
        _, top5_pred = clip_logits.topk(5, dim=1)
        local_top1 += (top5_pred[:, 0] == y).sum()
        local_top5 += (top5_pred == y.unsqueeze(1)).any(dim=1).sum()
        local_n += y.size(0)
    dist.all_reduce(local_top1, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_top5, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_n, op=dist.ReduceOp.SUM)
    return (local_top1 / local_n).item(), (local_top5 / local_n).item()


# ---- Train one epoch ------------------------------------------------------

def train_one_epoch_ta(
    model: DDP,
    loader: DataLoader,
    kd_loss: PerFrameKDLoss,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    global_step_start: int,
    rank: int,
    log_every: int = 50,
    max_steps: int | None = None,
) -> tuple[int, dict[str, float]]:
    model.train()
    steps_per_epoch = len(loader)
    running = {"loss": 0.0, "ce": 0.0, "kd": 0.0}
    running_correct = 0       # uses last-frame logits
    running_n = 0
    t_start = time.time()
    global_step = global_step_start

    for step, (x, y, soft) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        soft = soft.to(device, non_blocking=True)

        lr = lr_at_step(global_step, steps_per_epoch, warmup_epochs, total_epochs, base_lr)
        set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)                            # (B,T,K)
            out = kd_loss(logits, soft, y)               # dict
            loss = out["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred = logits[:, -1].argmax(dim=1)
            running_correct += (pred == y).sum().item()
            running_n += y.size(0)
            running["loss"] += float(loss.item()) * y.size(0)
            running["ce"] += float(out["ce"].item()) * y.size(0)
            running["kd"] += float(out["kd"].item()) * y.size(0)

        if (step + 1) % log_every == 0 and is_main(rank):
            elapsed = time.time() - t_start
            ips = running_n / max(elapsed, 1e-6)
            acc = running_correct / running_n
            avg_loss = running["loss"] / running_n
            avg_ce = running["ce"] / running_n
            avg_kd = running["kd"] / running_n
            print(
                f"  [ep {epoch+1:02d}] step {step+1:5d}/{steps_per_epoch}  "
                f"lr={lr:.2e}  loss={avg_loss:.4f}  ce={avg_ce:.4f}  kd={avg_kd:.4f}  "
                f"acc(t7)={acc:.4f}  ips={ips:.1f}  el={elapsed/60:.1f}m",
                flush=True,
            )
        global_step += 1
        if max_steps is not None and (step + 1) >= max_steps:
            if is_main(rank):
                print(f"[dry-run] max_steps={max_steps} reached, exiting epoch early", flush=True)
            break

    return global_step, {
        "loss": running["loss"] / max(running_n, 1),
        "ce": running["ce"] / max(running_n, 1),
        "kd": running["kd"] / max(running_n, 1),
        "acc": running_correct / max(running_n, 1),
    }


# ---- Main -----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64, help="per-GPU")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Plan: 5e-4 × world_size=2 = 1e-3")
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--n_div", type=int, default=8,
                        help="TSM channel-fold divisor (causal shift fraction = 1/n_div)")
    parser.add_argument(
        "--soft_labels", type=Path, default=DEFAULT_SOFT_LABELS,
        help="path to .npy from Day 5 (95.04% SWA+TTA preferred)",
    )
    parser.add_argument(
        "--ckpt_dir", type=Path,
        default=PROJECT_ROOT / "models" / "teacher_assistant",
    )
    parser.add_argument(
        "--log_dir", type=Path, default=PROJECT_ROOT / "logs",
    )
    parser.add_argument("--ckpt_every", type=int, default=1,
                        help="per-epoch save (Day 5 learning: snapshots needed for late-epoch SWA option)")
    parser.add_argument("--job_id", type=str, default=os.environ.get("SLURM_JOB_ID", "local"))
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="dry-run: exit after N steps per epoch (default: full epoch)")
    args = parser.parse_args()

    # Fallback to baseline soft labels if SWA+TTA missing
    if not args.soft_labels.is_file():
        if FALLBACK_SOFT_LABELS.is_file():
            args.soft_labels = FALLBACK_SOFT_LABELS
        else:
            raise FileNotFoundError(
                f"Neither {DEFAULT_SOFT_LABELS} nor {FALLBACK_SOFT_LABELS} found"
            )

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if is_main(rank):
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        args.log_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    if is_main(rank):
        print(f"[ddp] world_size={world_size}  rank={rank}  local_rank={local_rank}", flush=True)
        print(f"[args] {vars(args)}", flush=True)
        print(f"[kd] soft_labels = {args.soft_labels}", flush=True)

    # ---- Data ----
    train_ds = JesterH5KD(
        split="train",
        soft_labels_path=args.soft_labels,
        T=args.T, img_size=args.img_size, training=True,
    )
    val_ds = JesterH5Torch(
        split="validation", T=args.T, img_size=args.img_size, training=False,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init,
    )
    log(rank, f"[data] train: {len(train_ds)} clips, {len(train_loader)} steps/epoch/rank")
    log(rank, f"[data] val:   {len(val_ds)} clips, {len(val_loader)} steps/rank")

    # ---- Model ----
    latest_path = args.ckpt_dir / "latest.pt"
    cold_start = not latest_path.is_file()
    model = TaTSMMobileNetV2(
        n_classes=27, n_segment=args.T, n_div=args.n_div,
        dropout=0.5, pretrained=cold_start,    # only download on cold start
    ).to(device)
    if not cold_start:
        log(rank, "[init] checkpoint detected: skipping ImageNet init (will load via maybe_resume)")
    else:
        log(rank, "[init] cold start: loaded MobileNetV2 ImageNet pretrained backbone")

    # SyncBN BEFORE DDP wrap (required ordering)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    if is_main(rank):
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] params={n_params/1e6:.3f}M  TSM sites={model.module.n_tsm_sites}", flush=True)

    # ---- KD loss + optimizer/scaler ----
    kd_loss = PerFrameKDLoss(
        n_classes=27, temperature=args.temperature, alpha=args.alpha,
        frame_weights=DEFAULT_FRAME_WEIGHTS,
    ).to(device)
    optimizer = build_optimizer_ta(model.module, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # ---- Resume ----
    start_epoch, best_val_acc = maybe_resume(model, optimizer, scaler, latest_path, device, rank)

    # ---- Train loop ----
    steps_per_epoch = len(train_loader)
    global_step = start_epoch * steps_per_epoch
    epoch_summary: list[dict] = []
    t_train_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        if is_main(rank):
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===", flush=True)
        t_ep_start = time.time()
        global_step, train_stats = train_one_epoch_ta(
            model, train_loader, kd_loss, optimizer, scaler, device,
            epoch, args.epochs, args.warmup_epochs, args.lr,
            global_step, rank,
            max_steps=args.max_train_steps,
        )
        t_train = time.time() - t_ep_start

        t_val_start = time.time()
        val_top1, val_top5 = evaluate_ta_ddp(model, val_loader, device, last_frame_only=True)
        t_val = time.time() - t_val_start
        if val_top1 > best_val_acc:
            best_val_acc = val_top1

        log(rank,
            f"[ep {epoch+1:02d}] train_loss={train_stats['loss']:.4f} "
            f"ce={train_stats['ce']:.4f} kd={train_stats['kd']:.4f} "
            f"train_acc(t7)={train_stats['acc']:.4f}  "
            f"val_top1={val_top1:.4f} val_top5={val_top5:.4f}  "
            f"t_train={t_train/60:.1f}m t_val={t_val:.1f}s")

        if is_main(rank):
            rng_state = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            }
            save_ckpt(latest_path, model, optimizer, scaler, epoch, best_val_acc, rng_state)
            if (epoch + 1) % args.ckpt_every == 0 or (epoch + 1) == args.epochs:
                snap = args.ckpt_dir / f"epoch_{epoch+1:02d}.pt"
                save_ckpt(snap, model, optimizer, scaler, epoch, best_val_acc, rng_state)
                print(f"[ckpt] saved snapshot -> {snap.name}", flush=True)
            if val_top1 == best_val_acc:
                save_ckpt(args.ckpt_dir / "best.pt", model, optimizer, scaler, epoch, best_val_acc, rng_state)

            epoch_summary.append({
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_ce": train_stats["ce"],
                "train_kd": train_stats["kd"],
                "train_acc": train_stats["acc"],
                "val_top1": val_top1,
                "val_top5": val_top5,
                "best_val_acc": best_val_acc,
                "t_train_min": t_train / 60.0,
                "t_val_s": t_val,
            })
            json_path = args.log_dir / f"train_ta_{args.job_id}.json"
            with json_path.open("w") as f:
                json.dump({"args": {k: str(v) if isinstance(v, Path) else v
                                    for k, v in vars(args).items()},
                          "epochs": epoch_summary,
                          "world_size": world_size}, f, indent=2)
        dist.barrier()

    total_time = time.time() - t_train_start
    log(rank, f"\n[done] total wall = {total_time/3600:.2f}h  best_val_top1 = {best_val_acc:.4f}")
    cleanup_ddp()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
