#!/usr/bin/env python3
"""Train a larger Jester teacher and distill into an ai8x deployable student."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
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


class CachedJester(Dataset):
    """Cached uint8 Jester split with MAX78000-style [-1, 1] normalization."""

    def __init__(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.data = payload["data"]
        self.labels = payload["labels"].long()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.data[index].float().div(255.0).sub(0.5).mul(256.0).round()
        image = image.clamp(min=-128.0, max=127.0).div(128.0)
        return image, self.labels[index]


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TeacherBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


TEACHER_VARIANTS = {
    "big": {
        "channels": (64, 96, 128, 192),
        "blocks": (2, 2, 2, 2),
        "hidden": 256,
        "dropout": (0.25, 0.15),
    },
    "huge": {
        "channels": (96, 160, 224, 320),
        "blocks": (3, 3, 3, 3),
        "hidden": 512,
        "dropout": (0.35, 0.20),
    },
}


class ConfigurableJesterTeacher(nn.Module):
    """Software-only residual CNN teacher for 12-plane signed motion."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 27,
        channels: tuple[int, int, int, int] = (64, 96, 128, 192),
        blocks: tuple[int, int, int, int] = (2, 2, 2, 2),
        hidden: int = 256,
        dropout: tuple[float, float] = (0.25, 0.15),
    ) -> None:
        super().__init__()
        feature_layers: list[nn.Module] = []
        in_ch = input_channels
        for stage, (out_ch, num_blocks) in enumerate(zip(channels, blocks)):
            feature_layers.append(ConvBNReLU(in_ch, out_ch))
            feature_layers.extend(TeacherBlock(out_ch) for _ in range(num_blocks))
            if stage < len(channels) - 1:
                feature_layers.append(nn.MaxPool2d(2))
            else:
                feature_layers.append(nn.AdaptiveAvgPool2d(1))
            in_ch = out_ch

        self.features = nn.Sequential(*feature_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout[0]),
            nn.Linear(channels[-1], hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout[1]),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class BigJesterTeacher(ConfigurableJesterTeacher):
    """Compatibility wrapper for the original teacher architecture."""

    def __init__(self, input_channels: int = 12, num_classes: int = 27) -> None:
        super().__init__(input_channels=input_channels, num_classes=num_classes, **TEACHER_VARIANTS["big"])


def build_teacher(args, device: torch.device) -> nn.Module:
    config = TEACHER_VARIANTS[args.teacher_variant]
    teacher = ConfigurableJesterTeacher(num_classes=args.num_classes, **config)
    return teacher.to(device)


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    values: list[float] = []
    for k in topk:
        values.append(float(correct[:k].reshape(-1).float().sum().item() * 100.0 / target.numel()))
    return values


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            top1, top5 = topk_accuracy(logits, labels)
            batch = labels.numel()
            loss_sum += float(loss.item()) * batch
            top1_sum += top1 * batch
            top5_sum += top5 * batch
            count += batch
    return top1_sum / count, top5_sum / count, loss_sum / count


def save_checkpoint(path: Path, model: nn.Module, epoch: int, top1: float, top5: float, arch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "arch": arch,
            "extras": {"top1": top1, "top5": top5},
        },
        path,
    )


def load_state_dict_flexible(model: nn.Module, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    state_dict = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"loaded {path}", flush=True)
    print(f"  missing_keys={len(missing)} unexpected_keys={len(unexpected)}", flush=True)
    if missing:
        print(f"  first_missing={missing[:8]}", flush=True)
    if unexpected:
        print(f"  first_unexpected={unexpected[:8]}", flush=True)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_teacher(args, train_loader, val_loader, device, amp: bool, out_dir: Path) -> Path:
    teacher = build_teacher(args, device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.teacher_epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_top1 = -math.inf
    best_path = out_dir / "teacher_best.pth.tar"

    print(f"teacher_params={count_parameters(teacher)}", flush=True)
    for epoch in range(args.teacher_epochs):
        teacher.train()
        start = time.time()
        loss_sum = 0.0
        top1_sum = 0.0
        seen = 0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = teacher(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            top1 = topk_accuracy(logits.detach(), labels, (1,))[0]
            batch = labels.numel()
            loss_sum += float(loss.item()) * batch
            top1_sum += top1 * batch
            seen += batch
            if step == 1 or step % args.print_freq == 0 or step == len(train_loader):
                print(
                    f"teacher epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss_sum / seen:.4f} top1={top1_sum / seen:.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}",
                    flush=True,
                )
        scheduler.step()
        val_top1, val_top5, val_loss = evaluate(teacher, val_loader, device, amp)
        is_best = val_top1 > best_top1
        if is_best:
            best_top1 = val_top1
            save_checkpoint(best_path, teacher, epoch, val_top1, val_top5, f"{args.teacher_variant}_jester_teacher")
        print(
            f"TEACHER epoch={epoch} val_top1={val_top1:.3f} val_top5={val_top5:.3f} "
            f"val_loss={val_loss:.4f} best_top1={best_top1:.3f} "
            f"time={time.time() - start:.1f}s",
            flush=True,
        )
    return best_path


def build_student(args, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(Path(args.ai8x_training).resolve()))
    import ai8x  # pylint: disable=import-outside-toplevel
    from models.jestermotion12resxconv import jestermotion12resxconv  # pylint: disable=import-outside-toplevel

    ai8x.set_device(85, False, False, verbose=False)
    student = jestermotion12resxconv(num_classes=args.num_classes, num_channels=12, dimensions=(64, 64), bias=True)
    return student.to(device)


def train_student_distill(args, teacher_path: Path, train_loader, val_loader, device, amp: bool, out_dir: Path) -> Path:
    teacher = build_teacher(args, device)
    load_state_dict_flexible(teacher, teacher_path, device)
    teacher.eval()

    student = build_student(args, device)
    if args.student_init:
        init_path = Path(args.student_init)
        if init_path.is_file():
            load_state_dict_flexible(student, init_path, device)
        else:
            print(f"student_init_missing={init_path}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.student_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.student_epochs, 1))
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_top1 = -math.inf
    best_path = out_dir / "student_distilled_best.pth.tar"

    print(f"student_params={count_parameters(student)}", flush=True)
    for epoch in range(args.student_epochs):
        student.train()
        start = time.time()
        loss_sum = 0.0
        ce_sum = 0.0
        kd_sum = 0.0
        top1_sum = 0.0
        seen = 0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp):
                    teacher_logits = teacher(images)
            with torch.cuda.amp.autocast(enabled=amp):
                student_logits = student(images)
                ce_loss = ce(student_logits, labels)
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / args.temperature, dim=1),
                    F.softmax(teacher_logits / args.temperature, dim=1),
                    reduction="batchmean",
                ) * (args.temperature * args.temperature)
                loss = (1.0 - args.distill_alpha) * ce_loss + args.distill_alpha * kd_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            top1 = topk_accuracy(student_logits.detach(), labels, (1,))[0]
            batch = labels.numel()
            loss_sum += float(loss.item()) * batch
            ce_sum += float(ce_loss.item()) * batch
            kd_sum += float(kd_loss.item()) * batch
            top1_sum += top1 * batch
            seen += batch
            if step == 1 or step % args.print_freq == 0 or step == len(train_loader):
                print(
                    f"student epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss_sum / seen:.4f} ce={ce_sum / seen:.4f} "
                    f"kd={kd_sum / seen:.4f} top1={top1_sum / seen:.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}",
                    flush=True,
                )
        scheduler.step()
        val_top1, val_top5, val_loss = evaluate(student, val_loader, device, amp)
        is_best = val_top1 > best_top1
        if is_best:
            best_top1 = val_top1
            save_checkpoint(best_path, student, epoch, val_top1, val_top5, "jestermotion12resxconv")
        print(
            f"STUDENT epoch={epoch} val_top1={val_top1:.3f} val_top5={val_top5:.3f} "
            f"val_loss={val_loss:.4f} best_top1={best_top1:.3f} "
            f"time={time.time() - start:.1f}s",
            flush=True,
        )
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--ai8x-training", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--student-init", default="")
    parser.add_argument("--num-classes", type=int, default=27)
    parser.add_argument("--teacher-variant", choices=sorted(TEACHER_VARIANTS), default="big")
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--student-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--teacher-lr", type=float, default=3e-4)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--distill-alpha", type=float, default=0.7)
    parser.add_argument("--print-freq", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    cache_dir = Path(args.cache_dir)
    train_dataset = CachedJester(cache_dir / "train.pt")
    val_dataset = CachedJester(cache_dir / "val.pt")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if (cache_dir / "labels.json").is_file():
        shutil.copy2(cache_dir / "labels.json", out_dir / "labels.json")
    if (cache_dir / "cache_config.json").is_file():
        shutil.copy2(cache_dir / "cache_config.json", out_dir / "cache_config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(device.type == "cuda" and not args.no_amp)
    print(f"device={device} amp={amp}", flush=True)
    print(f"train={len(train_dataset)} val={len(val_dataset)} batch_size={args.batch_size}", flush=True)

    if args.sanity_only:
        teacher = build_teacher(args, device)
        student = build_student(args, device)
        images, labels = next(iter(train_loader))
        images = images.to(device)
        with torch.no_grad():
            print(f"sanity_teacher_logits={tuple(teacher(images).shape)}", flush=True)
            print(f"sanity_student_logits={tuple(student(images).shape)} labels={tuple(labels.shape)}", flush=True)
        return

    teacher_path = train_teacher(args, train_loader, val_loader, device, amp, out_dir)
    student_path = train_student_distill(args, teacher_path, train_loader, val_loader, device, amp, out_dir)
    print(f"DISTILL_DONE teacher_best={teacher_path} student_best={student_path}", flush=True)


if __name__ == "__main__":
    main()
