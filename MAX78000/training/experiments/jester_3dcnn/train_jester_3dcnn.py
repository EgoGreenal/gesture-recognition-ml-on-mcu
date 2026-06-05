#!/usr/bin/env python3
"""Train true Conv3D Jester classifiers and run software weight PTQ.

This script is intentionally separate from the ai8x deployable models. MAX78000
ai8x synthesis does not expose a Conv3d hardware operator, so the quantized
checkpoint produced here is for software/accuracy probing only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CachedJester3D(Dataset):
    """Cached uint8 Jester split exposed as N x C x T x H x W tensors."""

    def __init__(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.data = payload["data"]
        self.labels = payload["labels"].long()
        self.frames = int(payload.get("frames", self.data.shape[1]))
        self.image_size = int(payload.get("image_size", self.data.shape[-1]))
        self.encoding = str(payload.get("encoding", "unknown"))

        if self.data.ndim != 4:
            raise ValueError(f"Expected cached data with shape N,T,H,W; got {tuple(self.data.shape)}")

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video = self.data[index].float()
        # Match the MAX78000-style signed 8-bit input range used in our ai8x runs.
        video = video.div(255.0).sub(0.5).mul(256.0).round().clamp(-128.0, 127.0).div(128.0)
        return video.unsqueeze(0), self.labels[index]


class ConvBNReLU3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class JesterLite3D(nn.Module):
    """Small true-3D CNN for server-side 8/12 frame experiments."""

    def __init__(
        self,
        num_classes: int,
        width_mult: float = 1.0,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        channels = [max(8, int(round(c * width_mult))) for c in (16, 32, 64, 96)]
        layers: list[nn.Module] = [
            ConvBNReLU3D(1, channels[0]),
            ResidualBlock3D(channels[0]),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            ConvBNReLU3D(channels[0], channels[1]),
            ResidualBlock3D(channels[1]),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            ConvBNReLU3D(channels[1], channels[2]),
            ResidualBlock3D(channels[2]),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            ConvBNReLU3D(channels[2], channels[3]),
            ResidualBlock3D(channels[3]),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    values: list[float] = []
    for k in topk:
        values.append(float(correct[:k].reshape(-1).float().sum().item() * 100.0 / target.numel()))
    return values


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    count = 0
    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(videos)
            loss = F.cross_entropy(logits, labels)
        top1, top5 = topk_accuracy(logits, labels, (1, 5))
        batch = labels.numel()
        loss_sum += float(loss.item()) * batch
        top1_sum += top1 * batch
        top5_sum += top5 * batch
        count += batch
    return top1_sum / count, top5_sum / count, loss_sum / count


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    epoch: int,
    print_freq: int,
) -> tuple[float, float]:
    model.train()
    start = time.time()
    loss_sum = 0.0
    top1_sum = 0.0
    seen = 0
    for step, (videos, labels) in enumerate(loader, start=1):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(videos)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        top1 = topk_accuracy(logits.detach(), labels, (1,))[0]
        batch = labels.numel()
        loss_sum += float(loss.item()) * batch
        top1_sum += top1 * batch
        seen += batch
        if step == 1 or step % print_freq == 0 or step == len(loader):
            elapsed = time.time() - start
            print(
                f"train epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss_sum / seen:.4f} top1={top1_sum / seen:.3f} "
                f"ips={seen / max(elapsed, 1e-6):.1f} lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )
    return loss_sum / seen, top1_sum / seen


def quantize_tensor_symmetric_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_abs = tensor.detach().abs().max()
    if float(max_abs) == 0.0:
        scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
    else:
        scale = max_abs / 127.0
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    dequantized = quantized.to(tensor.dtype) * scale
    return quantized.cpu(), dequantized.to(tensor.device), scale.detach().cpu()


def make_weight_ptq_model(model: nn.Module) -> tuple[nn.Module, dict[str, dict[str, float]]]:
    quantized_model = copy.deepcopy(model)
    stats: dict[str, dict[str, float]] = {}
    for name, module in quantized_model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            q_weight, dq_weight, w_scale = quantize_tensor_symmetric_int8(module.weight.data)
            module.weight.data.copy_(dq_weight)
            stats[f"{name}.weight"] = {
                "scale": float(w_scale),
                "numel": int(q_weight.numel()),
                "min_q": int(q_weight.min()),
                "max_q": int(q_weight.max()),
            }
            if module.bias is not None:
                q_bias, dq_bias, b_scale = quantize_tensor_symmetric_int8(module.bias.data)
                module.bias.data.copy_(dq_bias)
                stats[f"{name}.bias"] = {
                    "scale": float(b_scale),
                    "numel": int(q_bias.numel()),
                    "min_q": int(q_bias.min()),
                    "max_q": int(q_bias.max()),
                }
    return quantized_model, stats


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    top1: float,
    top5: float,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "arch": "jester_lite_3d",
            "args": vars(args),
            "extras": {
                "top1": top1,
                "top5": top5,
                "note": "True Conv3D software model; not directly supported by MAX78000 ai8x synthesis.",
            },
            "history": history,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    labels_path = cache_dir / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.is_file() else []
    if labels_path.is_file():
        shutil.copy2(labels_path, out_dir / "labels.json")
    if (cache_dir / "cache_config.json").is_file():
        shutil.copy2(cache_dir / "cache_config.json", out_dir / "cache_config.json")

    train_dataset = CachedJester3D(cache_dir / "train.pt")
    val_dataset = CachedJester3D(cache_dir / "val.pt")
    num_classes = len(labels) if labels else int(train_dataset.labels.max().item()) + 1
    if args.frames and args.frames != train_dataset.frames:
        raise ValueError(f"--frames={args.frames} does not match cache frames={train_dataset.frames}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(device.type == "cuda" and not args.no_amp)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = JesterLite3D(num_classes=num_classes, width_mult=args.width_mult, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    print(
        f"device={device} amp={amp} train={len(train_dataset)} val={len(val_dataset)} "
        f"frames={train_dataset.frames} size={train_dataset.image_size} encoding={train_dataset.encoding} "
        f"classes={num_classes} batch_size={args.batch_size}",
        flush=True,
    )
    print(f"parameters={count_parameters(model)}", flush=True)
    print("max78000_support=false reason=ai8x/MAX78000 has no Conv3d hardware/YAML operator", flush=True)

    if args.sanity_only:
        videos, labels_batch = next(iter(train_loader))
        videos = videos.to(device)
        with torch.no_grad():
            logits = model(videos)
        print(
            f"sanity videos={tuple(videos.shape)} logits={tuple(logits.shape)} "
            f"labels={tuple(labels_batch.shape)}",
            flush=True,
        )
        return

    best_top1 = -math.inf
    history: list[dict[str, float]] = []
    best_path = out_dir / "best.pth.tar"
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_top1 = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, amp, epoch, args.print_freq
        )
        val_top1, val_top5, val_loss = evaluate(model, val_loader, device, amp)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        print("summary " + json.dumps(row), flush=True)
        save_checkpoint(out_dir / "last.pth.tar", model, args, epoch, val_top1, val_top5, history)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_top1 > best_top1:
            best_top1 = val_top1
            save_checkpoint(best_path, model, args, epoch, val_top1, val_top5, history)
            print(f"saved_best epoch={epoch} val_top1={val_top1:.3f} val_top5={val_top5:.3f}", flush=True)

    quantized_model, quant_stats = make_weight_ptq_model(model)
    q_top1, q_top5, q_loss = evaluate(quantized_model, val_loader, device, amp)
    save_checkpoint(out_dir / "ptq_int8_weight_best.pth.tar", quantized_model, args, args.epochs, q_top1, q_top5, history)
    (out_dir / "ptq_int8_weight_stats.json").write_text(json.dumps(quant_stats, indent=2), encoding="utf-8")
    summary = {
        "model": "jester_lite_3d",
        "frames": train_dataset.frames,
        "encoding": train_dataset.encoding,
        "image_size": train_dataset.image_size,
        "num_classes": num_classes,
        "parameters": count_parameters(model),
        "best_val_top1": best_top1,
        "final_val_top1": history[-1]["val_top1"],
        "final_val_top5": history[-1]["val_top5"],
        "ptq_int8_weight_val_top1": q_top1,
        "ptq_int8_weight_val_top5": q_top5,
        "ptq_int8_weight_val_loss": q_loss,
        "max78000_ai8x_conv3d_supported": False,
        "max78000_note": "Conv3D is a software-only experiment here; ai8x synthesis supports Conv1d/Conv2d/ConvTranspose2d/Linear but not Conv3d.",
        "best_checkpoint": str(best_path),
        "ptq_checkpoint": str(out_dir / "ptq_int8_weight_best.pth.tar"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps(summary), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a true Conv3D Jester model and software PTQ probe")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--print-freq", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
