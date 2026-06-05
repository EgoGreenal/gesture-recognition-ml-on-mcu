"""Day 5 path-B: fine-tune from best.pt with constant low LR + per-epoch ckpt for proper SWA.

Goal: push val_top1 from 94.71% past 95%.

Design:
  * Resume from best.pt (epoch 28, val=94.71%)
  * Constant LR = 2e-5 (10× lower than cosine peak 2e-4) — small enough to refine without
    disrupting converged state
  * No warmup (already at converged point)
  * 5 fine-tune epochs
  * Save ckpt EVERY epoch (not every 5) → 5 snapshots for proper SWA
  * Reuse all DDP / data / AMP plumbing from train_super_teacher.py

Output:
  models/super_teacher_ft/ft_ep_{1..5}.pt   — fine-tune snapshots
  models/super_teacher_ft/latest.pt          — last fine-tune state
  logs/fine_tune_<jobid>.json                — train+val per-epoch metrics
"""

from __future__ import annotations

import argparse
import json
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
from swin_t_video import SuperTeacherSwinT
from train_super_teacher import (
    setup_ddp, cleanup_ddp, is_main, log,
    build_optimizer, evaluate_ddp,
)


PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")


def save_model_only(path: Path, model: DDP, epoch_idx: int) -> None:
    """Lightweight ckpt save (just model state, no optimizer/scaler)."""
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.module.state_dict(), "epoch": epoch_idx}, tmp)
    tmp.rename(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_ckpt", type=Path,
                        default=PROJECT_ROOT / "models/super_teacher/best.pt")
    parser.add_argument("--out_dir", type=Path,
                        default=PROJECT_ROOT / "models/super_teacher_ft")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--job_id", type=str, default=os.environ.get("SLURM_JOB_ID", "local"))
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if is_main(rank):
        args.out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    log(rank, f"[ft] ddp world_size={world_size}  src={args.src_ckpt}  lr={args.lr}  epochs={args.epochs}")

    # Data
    train_ds = JesterH5Torch(split="train", T=args.T, img_size=args.img_size, training=True)
    val_ds = JesterH5Torch(split="validation", T=args.T, img_size=args.img_size, training=False)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init)
    log(rank, f"[ft] train: {len(train_ds)} clips, {len(train_loader)} steps/epoch/rank")

    # Model + load from best.pt
    model = SuperTeacherSwinT(n_classes=27, pretrained_2d=None).to(device)
    payload = torch.load(args.src_ckpt, map_location=device)
    sd = payload["model"] if "model" in payload else payload
    model.load_state_dict(sd)
    log(rank, f"[ft] loaded {args.src_ckpt} (orig epoch={payload.get('epoch','?')}, "
              f"orig best_val={payload.get('best_val_acc','?')})")
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Optimizer (constant LR, no schedule)
    optimizer = build_optimizer(model.module, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    crit = nn.CrossEntropyLoss()

    # Pre-FT val baseline (sanity check)
    val_top1, val_top5 = evaluate_ddp(model, val_loader, device, world_size)
    log(rank, f"[ft] PRE val_top1={val_top1:.4f} val_top5={val_top5:.4f}")

    epoch_summary: list[dict] = []
    t_total = time.time()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # Train one epoch
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_n = 0
        t_ep = time.time()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                running_correct += (pred == y).sum().item()
                running_n += y.size(0)
                running_loss += loss.item() * y.size(0)
            if (step + 1) % 200 == 0 and is_main(rank):
                el = time.time() - t_ep
                print(f"  [ft ep {epoch+1}] step {step+1}/{len(train_loader)}  "
                      f"loss={running_loss/running_n:.4f}  acc={running_correct/running_n:.4f}  "
                      f"el={el/60:.1f}m", flush=True)
        t_train = time.time() - t_ep
        train_loss = running_loss / max(running_n, 1)
        train_acc = running_correct / max(running_n, 1)

        # Val
        val_top1, val_top5 = evaluate_ddp(model, val_loader, device, world_size)
        log(rank, f"[ft ep {epoch+1}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_top1={val_top1:.4f} val_top5={val_top5:.4f}  t_train={t_train/60:.1f}m")

        # Save snapshot (rank 0)
        if is_main(rank):
            snap = args.out_dir / f"epoch_{epoch+1:02d}.pt"
            save_model_only(snap, model, epoch + 1)
            save_model_only(args.out_dir / "latest.pt", model, epoch + 1)
            epoch_summary.append({
                "ft_epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_top1": val_top1,
                "val_top5": val_top5,
                "t_train_min": t_train / 60.0,
            })
            with (PROJECT_ROOT / "logs" / f"fine_tune_{args.job_id}.json").open("w") as f:
                json.dump({"args": {k: str(v) if isinstance(v, Path) else v
                                    for k, v in vars(args).items()},
                          "world_size": world_size,
                          "epochs": epoch_summary}, f, indent=2)
        dist.barrier()

    log(rank, f"[ft done] total={time.time()-t_total:.0f}s = {(time.time()-t_total)/60:.1f}m")
    cleanup_ddp()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
