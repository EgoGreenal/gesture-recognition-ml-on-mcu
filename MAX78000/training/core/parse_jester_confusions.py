#!/usr/bin/env python3
"""Parse ai8x-training confusion logs and render comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def parse_log_spec(spec: str) -> tuple[str, Path]:
    name, path = spec.split("=", 1)
    return name, Path(path)


def parse_confusion_matrix(path: Path, classes: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "==> Confusion:"
    index = text.rfind(marker)
    if index < 0:
        raise ValueError(f"No confusion matrix found in {path}")

    rows: list[list[float]] = []
    for line in text[index + len(marker):].splitlines():
        if len(rows) >= classes:
            break
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if "[" not in stripped and rows:
            break
        nums = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", stripped)]
        if nums:
            rows.append(nums)

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (classes, classes):
        raise ValueError(f"Expected {(classes, classes)} confusion matrix in {path}, got {matrix.shape}")
    return matrix


def write_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def plot_matrix(path: Path, title: str, labels: list[str], matrix: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, ax = plt.subplots(figsize=(13, 11), dpi=180)
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_overview(path: Path, labels: list[str], matrices: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    names = list(matrices)
    fig, axes = plt.subplots(1, len(names), figsize=(22, 7), dpi=180, sharex=True, sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        im = ax.imshow(matrices[name], interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(range(len(labels)), fontsize=6)
        ax.set_yticklabels(range(len(labels)), fontsize=6)
    axes[0].set_ylabel("True")
    fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def safe_normalize(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denom, out=np.zeros_like(matrix, dtype=np.float64), where=denom != 0)


def write_summary(out: Path, labels: list[str], raw: dict[str, np.ndarray], norm: dict[str, np.ndarray]) -> None:
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "samples", "top1_percent", "mean_class_accuracy_percent"])
        for name, matrix in raw.items():
            samples = int(matrix.sum())
            top1 = 100.0 * float(np.trace(matrix)) / float(matrix.sum())
            mean_class = 100.0 * float(np.mean(np.diag(norm[name])))
            writer.writerow([name, samples, f"{top1:.6f}", f"{mean_class:.6f}"])

    with (out / "per_class_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "label", *[f"{name}_accuracy_percent" for name in raw]])
        for class_id, label in enumerate(labels):
            writer.writerow(
                [
                    class_id,
                    label,
                    *[f"{100.0 * norm[name][class_id, class_id]:.6f}" for name in raw],
                ]
            )

    with (out / "top_confusions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "true_id", "true_label", "pred_id", "pred_label", "count", "row_percent"])
        for name, matrix in raw.items():
            items = []
            for true_id in range(len(labels)):
                for pred_id in range(len(labels)):
                    if true_id == pred_id or matrix[true_id, pred_id] <= 0:
                        continue
                    items.append(
                        (
                            norm[name][true_id, pred_id],
                            matrix[true_id, pred_id],
                            true_id,
                            pred_id,
                        )
                    )
            for rate, count, true_id, pred_id in sorted(items, reverse=True)[:30]:
                writer.writerow(
                    [
                        name,
                        true_id,
                        labels[true_id],
                        pred_id,
                        labels[pred_id],
                        int(count),
                        f"{100.0 * rate:.6f}",
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--log", action="append", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    classes = len(labels)

    raw: dict[str, np.ndarray] = {}
    norm: dict[str, np.ndarray] = {}
    for spec in args.log:
        name, path = parse_log_spec(spec)
        raw[name] = parse_confusion_matrix(path, classes)
        norm[name] = safe_normalize(raw[name])
        write_matrix_csv(out / f"{name}_confusion_counts.csv", labels, raw[name])
        write_matrix_csv(out / f"{name}_confusion_normalized.csv", labels, norm[name])
        plot_matrix(out / f"{name}_confusion_normalized.png", name, labels, norm[name])

    plot_overview(out / "confusion_overview_normalized.png", labels, norm)
    write_summary(out, labels, raw, norm)


if __name__ == "__main__":
    main()
