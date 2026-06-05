"""Day 3-5: Train Super Teacher (Video Swin-T) on Jester with DDP.

Launch with torchrun:
  torchrun --standalone --nproc_per_node=4 train_super_teacher.py [args]

DDP wiring (critical):
  * `torch.distributed.init_process_group("nccl")` — NCCL backend, GPU collective ops
  * Each rank binds to one GPU via `torch.cuda.set_device(LOCAL_RANK)`
  * Model wrapped in `DistributedDataParallel(model, device_ids=[LOCAL_RANK])`
  * Train DataLoader uses `DistributedSampler(dataset, shuffle=True)`
  * **MUST call `train_sampler.set_epoch(epoch)` each epoch** — otherwise every
    epoch sees identical sample order (DistributedSampler RNG is seeded by epoch,
    not call count). Skipping this is the #1 silent DDP bug.
  * Val sampler uses `shuffle=False, drop_last=False` — full coverage
  * Val metrics: each rank computes local correct/total → `dist.all_reduce(SUM)` → global
  * Checkpoint save: rank 0 only; `dist.barrier()` sync before/after

Plan (Section 4 Day 3-5):
  * Pipeline H (112x112 RGB, 8 frames)
  * SwinTransformer3D + I3D head, 27 classes, init from Swin-T 2D ImageNet
  * batch 16 per-GPU * 4 GPU = 64 global; lr scaled 1e-4 -> 4e-4 (linear)
  * AdamW (wd 0.02, no decay on pos_embed / norm / relative_position_bias)
  * Cosine schedule + 3-epoch linear warmup
  * AMP (torch.amp.autocast + GradScaler)
  * 30 epoch; save ckpt every 5 epochs + always "latest"
  * Accept >= 96% val top-1 (Plan); 95-96% acceptable w/ risk-log note
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
from swin_t_video import SuperTeacherSwinT


PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
DEFAULT_2D_CKPT = (
    PROJECT_ROOT / "cache/torch/hub/checkpoints/swin_tiny_patch4_window7_224.pth"
)


# ---- DDP helpers -----------------------------------------------------------

def setup_ddp() -> tuple[int, int, int]:
    """Initialize NCCL group; returns (rank, local_rank, world_size)."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str) -> None:
    if is_main(rank):
        print(msg, flush=True)


# ---- Optimizer with paramwise weight-decay overrides -----------------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """AdamW with no-decay on norm / pos_embed / relative_position_bias_table.

    Matches mmaction swin config behavior. Backbone LR-mult is *not* applied
    (Plan uses uniform lr).
    """
    decay_params = []
    no_decay_params = []
    no_decay_keywords = ("relative_position_bias_table", "norm", "absolute_pos_embed")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in no_decay_keywords):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


# ---- LR schedule: linear warmup + cosine annealing -------------------------

def lr_at_step(step: int, steps_per_epoch: int, warmup_epochs: int,
               total_epochs: int, base_lr: float, min_lr: float = 0.0) -> float:
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---- Checkpoint -----------------------------------------------------------

def save_ckpt(path: Path, model: DDP, optimizer, scaler: GradScaler,
              epoch: int, best_val_acc: float, rng_state: dict) -> None:
    """Save full training state (only rank 0 should call this)."""
    payload = {
        "epoch": epoch,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_acc": best_val_acc,
        "rng_state": rng_state,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)


def maybe_resume(model: DDP, optimizer, scaler: GradScaler,
                 latest_path: Path, device: torch.device, rank: int) -> tuple[int, float]:
    """Load checkpoint if present; broadcast resumed state across ranks.

    Returns (start_epoch, best_val_acc). All ranks load identical state.
    """
    if not latest_path.is_file():
        return 0, 0.0
    log(rank, f"[resume] loading {latest_path}")
    payload = torch.load(latest_path, map_location=device)
    model.module.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    start_epoch = payload["epoch"] + 1
    best_val_acc = float(payload["best_val_acc"])
    # No need to manually broadcast — all ranks load the same file. Sync barrier
    # is what keeps them aligned.
    dist.barrier()
    log(rank, f"[resume] resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
    return start_epoch, best_val_acc


# ---- Eval (distributed) ----------------------------------------------------

@torch.no_grad()
def evaluate_ddp(model: DDP, loader: DataLoader, device: torch.device,
                 world_size: int) -> tuple[float, float]:
    """Run val on each rank's shard; all_reduce sums → global top1 / top5."""
    model.eval()
    local_top1 = torch.zeros(1, device=device)
    local_top5 = torch.zeros(1, device=device)
    local_n = torch.zeros(1, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
        _, top5_pred = logits.topk(5, dim=1)
        local_top1 += (top5_pred[:, 0] == y).sum()
        local_top5 += (top5_pred == y.unsqueeze(1)).any(dim=1).sum()
        local_n += y.size(0)
    dist.all_reduce(local_top1, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_top5, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_n, op=dist.ReduceOp.SUM)
    top1 = (local_top1 / local_n).item()
    top5 = (local_top5 / local_n).item()
    return top1, top5


# ---- Train one epoch -------------------------------------------------------

def train_one_epoch(
    model: DDP, loader: DataLoader, optimizer, scaler: GradScaler,
    device: torch.device, epoch: int, total_epochs: int, warmup_epochs: int,
    base_lr: float, global_step_start: int, rank: int, log_every: int = 50,
) -> tuple[int, float, float]:
    """Returns (next_global_step, avg_loss, avg_acc)."""
    model.train()
    crit = nn.CrossEntropyLoss()
    steps_per_epoch = len(loader)
    running_loss = 0.0
    running_correct = 0
    running_n = 0
    t_start = time.time()
    global_step = global_step_start

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        lr = lr_at_step(global_step, steps_per_epoch, warmup_epochs, total_epochs, base_lr)
        set_lr(optimizer, lr)

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

        if (step + 1) % log_every == 0 and is_main(rank):
            elapsed = time.time() - t_start
            ips = running_n / max(elapsed, 1e-6)
            avg_loss = running_loss / running_n
            acc = running_correct / running_n
            print(
                f"  [ep {epoch+1:02d}] step {step+1:5d}/{steps_per_epoch}  "
                f"lr={lr:.2e}  loss={avg_loss:.4f}  acc={acc:.4f}  "
                f"ips={ips:.1f}  el={elapsed/60:.1f}m",
                flush=True,
            )
        global_step += 1

    return global_step, running_loss / max(running_n, 1), running_correct / max(running_n, 1)


# ---- Main ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16, help="per-GPU")
    parser.add_argument("--lr", type=float, default=4e-4,
                        help="effective global LR for global_batch=64 (linear scale of Plan 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--ckpt_every", type=int, default=5)
    parser.add_argument(
        "--pretrained_2d", type=Path, default=DEFAULT_2D_CKPT,
        help="path to swin_tiny_patch4_window7_224.pth (only used at first start)",
    )
    parser.add_argument(
        "--ckpt_dir", type=Path,
        default=PROJECT_ROOT / "models" / "super_teacher",
    )
    parser.add_argument(
        "--log_dir", type=Path, default=PROJECT_ROOT / "logs",
    )
    parser.add_argument("--job_id", type=str, default=os.environ.get("SLURM_JOB_ID", "local"))
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if is_main(rank):
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        args.log_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    if is_main(rank):
        print(f"[ddp] world_size={world_size}  rank={rank}  local_rank={local_rank}", flush=True)
        print(f"[args] {vars(args)}", flush=True)

    # ---- Data ----
    train_ds = JesterH5Torch(split="train", T=args.T, img_size=args.img_size, training=True)
    val_ds = JesterH5Torch(split="validation", T=args.T, img_size=args.img_size, training=False)

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
    # NOTE: pretrained_2d only applies on cold start. On resume, model state
    # comes from checkpoint; we skip init_pretrained() in that case.
    latest_path = args.ckpt_dir / "latest.pt"
    cold_start = not latest_path.is_file()
    model = SuperTeacherSwinT(
        n_classes=27,
        dropout=0.5,
        drop_path_rate=args.drop_path_rate,
        pretrained_2d=str(args.pretrained_2d) if cold_start else None,
    ).to(device)
    if cold_start:
        log(rank, f"[init] cold start: loading 2D weights from {args.pretrained_2d}")
        model.init_pretrained()
    else:
        log(rank, f"[init] checkpoint detected: skipping 2D init (will load via maybe_resume)")
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    if is_main(rank):
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] params={n_params/1e6:.2f}M (expect ~28M)", flush=True)

    # ---- Optimizer / scaler ----
    optimizer = build_optimizer(model.module, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # ---- Resume (if applicable) ----
    start_epoch, best_val_acc = maybe_resume(model, optimizer, scaler, latest_path, device, rank)

    # ---- Train loop ----
    steps_per_epoch = len(train_loader)
    global_step = start_epoch * steps_per_epoch
    epoch_summary: list[dict] = []

    t_train_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)  # CRITICAL: re-seed sampler each epoch
        val_sampler.set_epoch(epoch)    # harmless for shuffle=False, kept for symmetry

        if is_main(rank):
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===", flush=True)
        t_ep_start = time.time()
        global_step, train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch, args.epochs, args.warmup_epochs, args.lr,
            global_step, rank,
        )
        t_train = time.time() - t_ep_start

        t_val_start = time.time()
        val_top1, val_top5 = evaluate_ddp(model, val_loader, device, world_size)
        t_val = time.time() - t_val_start
        if val_top1 > best_val_acc:
            best_val_acc = val_top1

        log(rank, f"[ep {epoch+1:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_top1={val_top1:.4f} val_top5={val_top5:.4f}  "
                  f"t_train={t_train/60:.1f}m t_val={t_val:.1f}s")

        # Save checkpoint (rank 0 only)
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
            # Also persist best
            if val_top1 == best_val_acc:
                best_path = args.ckpt_dir / "best.pt"
                save_ckpt(best_path, model, optimizer, scaler, epoch, best_val_acc, rng_state)

            epoch_summary.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_top1": val_top1,
                "val_top5": val_top5,
                "best_val_acc": best_val_acc,
                "t_train_min": t_train / 60.0,
                "t_val_s": t_val,
            })
            json_path = args.log_dir / f"super_teacher_{args.job_id}.json"
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
