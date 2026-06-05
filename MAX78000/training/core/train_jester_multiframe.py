#!/usr/bin/env python3
"""Train a compact multi-frame CNN on 20BN-Jester.

The model is intentionally simple: N grayscale frames are stacked as input
channels and processed by a 2D CNN. This keeps the first experiment close to a
future MAX78000 deployment path, where temporal information can be represented
as extra input channels.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    dataset_root: str
    annotations_dir: str
    output_dir: str
    frames: int
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    max_train_samples: int | None
    max_val_samples: int | None
    seed: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def split_annotation_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None
    for delimiter in (";", ",", "\t"):
        if delimiter in line:
            first, rest = line.split(delimiter, 1)
            return first.strip(), rest.strip()
    return None


def read_split(path: Path, label_to_idx: dict[str, int], videos_dir: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            parsed = split_annotation_line(raw_line)
            if parsed is None:
                continue
            video_id, label_name = parsed
            if label_name not in label_to_idx:
                continue
            video_dir = videos_dir / video_id
            if video_dir.is_dir():
                samples.append((video_dir, label_to_idx[label_name]))
    if not samples:
        raise ValueError(f"No usable labeled samples found in {path}")
    return samples


def limit_samples(samples: list[tuple[Path, int]], limit: int | None, seed: int) -> list[tuple[Path, int]]:
    if limit is None or limit <= 0 or limit >= len(samples):
        return samples
    rng = random.Random(seed)
    selected = list(samples)
    rng.shuffle(selected)
    return selected[:limit]


class JesterFrames(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        frames: int,
        image_size: int,
        train: bool,
    ) -> None:
        self.samples = samples
        self.frames = frames
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_dir, label = self.samples[index]
        frame_paths = sorted(video_dir.glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No frames in {video_dir}")

        if len(frame_paths) >= self.frames:
            indices = np.linspace(0, len(frame_paths) - 1, self.frames)
            indices = np.round(indices).astype(np.int64).tolist()
        else:
            indices = [round(i * (len(frame_paths) - 1) / max(self.frames - 1, 1)) for i in range(self.frames)]

        brightness = random.uniform(0.85, 1.15) if self.train else 1.0
        contrast = random.uniform(0.85, 1.15) if self.train else 1.0

        planes: list[np.ndarray] = []
        for frame_index in indices:
            image = Image.open(frame_paths[frame_index]).convert("L")
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            if self.train:
                image = ImageEnhance.Brightness(image).enhance(brightness)
                image = ImageEnhance.Contrast(image).enhance(contrast)
            array = np.asarray(image, dtype=np.float32) / 255.0
            planes.append(array)

        stacked = np.stack(planes, axis=0)
        stacked = (stacked - 0.5) / 0.5
        return torch.from_numpy(stacked), torch.tensor(label, dtype=torch.long)


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


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    prediction = output.argmax(dim=1)
    return (prediction == target).float().mean().item()


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
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    start = time.time()

    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_count += batch_size

        if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
            elapsed = time.time() - start
            print(
                f"{phase} epoch={epoch} batch={batch_index}/{len(loader)} "
                f"loss={total_loss / total_count:.4f} "
                f"acc={total_correct / total_count:.4f} "
                f"ips={total_count / max(elapsed, 1e-6):.1f}",
                flush=True,
            )

    return total_loss / total_count, total_correct / total_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact multi-frame CNN on 20BN-Jester")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-val-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(**vars(args))
    set_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    videos_dir = dataset_root / "20bn-jester-v1"
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(annotations_dir / "jester-v1-labels.csv")
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    train_samples = read_split(annotations_dir / "jester-v1-train.csv", label_to_idx, videos_dir)
    val_samples = read_split(annotations_dir / "jester-v1-validation.csv", label_to_idx, videos_dir)
    train_samples = limit_samples(train_samples, args.max_train_samples, args.seed)
    val_samples = limit_samples(val_samples, args.max_val_samples, args.seed + 1)

    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"classes={len(labels)} train={len(train_samples)} val={len(val_samples)}", flush=True)

    train_dataset = JesterFrames(train_samples, args.frames, args.image_size, train=True)
    val_dataset = JesterFrames(val_samples, args.frames, args.image_size, train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = TinyGestureNet(args.frames, len(labels)).to(device)
    print(model, flush=True)
    print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    print(f"device={device}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, epoch, "train")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, None, None, epoch, "val")
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print("summary", json.dumps(row), flush=True)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "labels": labels,
            "config": asdict(config),
            "history": history,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, output_dir / "last.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, output_dir / "best.pth")
            print(f"saved best val_acc={best_acc:.4f}", flush=True)

    print(f"done best_val_acc={best_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
