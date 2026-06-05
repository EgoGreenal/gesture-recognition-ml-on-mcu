#!/usr/bin/env python3
"""Cache and train a compact multi-frame CNN for 20BN-Jester.

This avoids random JPEG reads during training by first converting a selected
subset into contiguous uint8 tensor files under /tmp.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_annotation(path: Path, label_to_idx: dict[str, int], frames_root: Path) -> list[tuple[str, Path, int]]:
    samples: list[tuple[str, Path, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                video_id, label = line.split(";", 1)
            elif "," in line:
                video_id, label = line.split(",", 1)
            else:
                continue
            video_id = video_id.strip()
            label = label.strip()
            if label not in label_to_idx:
                continue
            video_dir = frames_root / video_id
            if video_dir.is_dir():
                samples.append((video_id, video_dir, label_to_idx[label]))
    return samples


def balanced_subset(
    samples: list[tuple[str, Path, int]],
    limit: int,
    seed: int,
) -> list[tuple[str, Path, int]]:
    if limit <= 0 or limit >= len(samples):
        return samples

    by_label: dict[int, list[tuple[str, Path, int]]] = defaultdict(list)
    for sample in samples:
        by_label[sample[2]].append(sample)

    rng = random.Random(seed)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    selected: list[tuple[str, Path, int]] = []
    label_ids = sorted(by_label)
    cursor = 0
    while len(selected) < limit and label_ids:
        label = label_ids[cursor % len(label_ids)]
        bucket = by_label[label]
        if bucket:
            selected.append(bucket.pop())
        else:
            label_ids.remove(label)
            cursor -= 1
        cursor += 1

    selected.sort(key=lambda item: int(item[0]) if item[0].isdigit() else item[0])
    return selected


def sample_frame_indices(count: int, frames: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [0] * frames
    return np.round(np.linspace(0, count - 1, frames)).astype(np.int64).tolist()


def read_video(video_dir: Path, frames: int, image_size: int) -> np.ndarray:
    frame_paths = sorted(video_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {video_dir}")

    planes = []
    for index in sample_frame_indices(len(frame_paths), frames):
        with Image.open(frame_paths[index]) as image:
            image = image.convert("L")
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            planes.append(np.asarray(image, dtype=np.uint8))
    return np.stack(planes, axis=0)


def build_split_cache(
    name: str,
    samples: list[tuple[str, Path, int]],
    output_path: Path,
    frames: int,
    image_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    skipped: list[tuple[str, str]] = []
    start = time.time()

    for index, (video_id, video_dir, label) in enumerate(samples, start=1):
        try:
            data.append(read_video(video_dir, frames, image_size))
            labels.append(label)
            ids.append(video_id)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            skipped.append((video_id, repr(exc)))

        if index == 1 or index % 100 == 0 or index == len(samples):
            elapsed = time.time() - start
            print(
                f"cache {name} {index}/{len(samples)} ok={len(data)} "
                f"skip={len(skipped)} ips={index / max(elapsed, 1e-6):.1f}",
                flush=True,
            )

    if not data:
        raise RuntimeError(f"No valid samples cached for {name}")

    tensor = torch.from_numpy(np.stack(data, axis=0))
    label_tensor = torch.tensor(labels, dtype=torch.long)
    torch.save(
        {
            "data": tensor,
            "labels": label_tensor,
            "ids": ids,
            "skipped": skipped,
            "frames": frames,
            "image_size": image_size,
        },
        output_path,
    )
    print(f"saved {name} cache {output_path} data={tuple(tensor.shape)} skipped={len(skipped)}", flush=True)


def build_cache(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dataset_root = Path(args.dataset_root)
    frames_root = dataset_root / "20bn-jester-v1"
    annotations = Path(args.annotations_dir)
    cache_dir = Path(args.cache_dir)
    labels = read_labels(annotations / "jester-v1-labels.csv")
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    train = parse_annotation(annotations / "jester-v1-train.csv", label_to_idx, frames_root)
    val = parse_annotation(annotations / "jester-v1-validation.csv", label_to_idx, frames_root)
    train = balanced_subset(train, args.train_samples, args.seed)
    val = balanced_subset(val, args.val_samples, args.seed + 1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (cache_dir / "cache_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"cache_dir={cache_dir}", flush=True)
    print(f"classes={len(labels)} train_selected={len(train)} val_selected={len(val)}", flush=True)
    build_split_cache("train", train, cache_dir / "train.pt", args.frames, args.image_size)
    build_split_cache("val", val, cache_dir / "val.pt", args.frames, args.image_size)


class CachedJester(Dataset):
    def __init__(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.data = payload["data"]
        self.labels = payload["labels"]

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.data[index].float().div(255.0).sub(0.5).div(0.5)
        return image, self.labels[index]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyGestureNet(nn.Module):
    def __init__(self, input_frames: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(input_frames, 16),
            nn.MaxPool2d(2),
            ConvBlock(16, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    phase: str,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    start = time.time()

    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                output = model(inputs)
                loss = criterion(output, targets)
            if optimizer is not None:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        count = inputs.size(0)
        total_loss += loss.item() * count
        total_correct += (output.argmax(dim=1) == targets).sum().item()
        total_count += count

        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            elapsed = time.time() - start
            print(
                f"{phase} epoch={epoch} batch={batch_index}/{len(loader)} "
                f"loss={total_loss / total_count:.4f} "
                f"acc={total_correct / total_count:.4f} "
                f"ips={total_count / max(elapsed, 1e-6):.1f}",
                flush=True,
            )

    return total_loss / total_count, total_correct / total_count


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = json.loads((cache_dir / "labels.json").read_text(encoding="utf-8"))
    train_dataset = CachedJester(cache_dir / "train.pt")
    val_dataset = CachedJester(cache_dir / "val.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = TinyGestureNet(args.frames, len(labels)).to(device)
    print(f"classes={len(labels)} train={len(train_dataset)} val={len(val_dataset)}", flush=True)
    print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    print(f"device={device}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_acc = -1.0
    epoch_offset = 0

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.resume_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.resume_history:
            history = list(checkpoint.get("history", []))
        best_acc = float(checkpoint.get("val_acc", -1.0))
        epoch_offset = int(checkpoint.get("epoch", 0)) if args.continue_epoch_numbers else 0
        print(
            f"resumed checkpoint={args.resume_checkpoint} "
            f"checkpoint_epoch={checkpoint.get('epoch')} initial_best_acc={best_acc:.4f}",
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        display_epoch = epoch_offset + epoch
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, display_epoch, "train")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, None, None, display_epoch, "val")
        scheduler.step()
        row = {
            "epoch": display_epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print("summary", json.dumps(row), flush=True)
        checkpoint = {
            "epoch": display_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "labels": labels,
            "history": history,
            "val_acc": val_acc,
            "frames": args.frames,
        }
        torch.save(checkpoint, output_dir / "last.pth")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, output_dir / "best.pth")
            print(f"saved best val_acc={best_acc:.4f}", flush=True)

    print(f"done best_val_acc={best_acc:.4f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache and train 20BN-Jester multi-frame CNN")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache")
    cache.add_argument("--dataset-root", required=True)
    cache.add_argument("--annotations-dir", required=True)
    cache.add_argument("--cache-dir", required=True)
    cache.add_argument("--frames", type=int, default=8)
    cache.add_argument("--image-size", type=int, default=64)
    cache.add_argument("--train-samples", type=int, default=5000)
    cache.add_argument("--val-samples", type=int, default=1000)
    cache.add_argument("--seed", type=int, default=7)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--cache-dir", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--frames", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--resume-checkpoint", default="")
    train_parser.add_argument("--resume-optimizer", action="store_true")
    train_parser.add_argument("--resume-history", action="store_true")
    train_parser.add_argument("--continue-epoch-numbers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "cache":
        build_cache(args)
    elif args.command == "train":
        train(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
