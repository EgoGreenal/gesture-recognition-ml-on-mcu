#!/usr/bin/env python3
"""Evaluate prefix-based early-exit policies for the Jester MAX78000 model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


DEFAULT_LABELS = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Down",
    "Swiping Up",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Pushing Two Fingers Away",
    "Pulling Two Fingers In",
    "Rolling Hand Forward",
    "Rolling Hand Backward",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Thumb Up",
    "Thumb Down",
    "Shaking Hand",
    "Stop Sign",
    "Drumming Fingers",
    "No gesture",
    "Doing other things",
]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai8x-training", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="jestermotion12resxconv",
        choices=("jestermotion12", "jestermotion12resxconv"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prefixes", default="4,6,8,10,12")
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--margins", default="0.00,0.05,0.10,0.15,0.20")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--class-threshold-grid", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--class-min-precision", default="0.76,0.78,0.80,0.82,0.85,0.90")
    parser.add_argument("--act-mode-8bit", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_state_dict(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    labels = checkpoint.get("extras", {}).get("labels") or DEFAULT_LABELS
    if len(labels) != len(DEFAULT_LABELS):
        labels = DEFAULT_LABELS
    return state_dict, labels


def load_model(args: argparse.Namespace, labels: list[str]):
    sys.path.insert(0, str(args.ai8x_training))

    import ai8x  # pylint: disable=import-outside-toplevel
    ai8x.set_device(85, args.act_mode_8bit, False, verbose=False)

    if args.model == "jestermotion12":
        from models.jestermotion12 import JesterMotion12Net  # pylint: disable=import-outside-toplevel

        model_cls = JesterMotion12Net
    elif args.model == "jestermotion12resxconv":
        from models.jestermotion12resxconv import (  # pylint: disable=import-outside-toplevel
            JesterMotion12ResXConvNet,
        )

        model_cls = JesterMotion12ResXConvNet
    else:
        raise ValueError(f"Unsupported model {args.model!r}")

    model = model_cls(
        num_classes=len(labels),
        dimensions=(64, 64),
        num_channels=12,
        bias=True,
    )
    state_dict, _ = load_state_dict(args.checkpoint)
    model.load_state_dict(state_dict)
    ai8x.update_model(model)
    return model, ai8x


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_prefixes(
    model,
    ai8x,
    data: torch.Tensor,
    targets: torch.Tensor,
    labels: list[str],
    prefixes: list[int],
    batch_size: int,
    act_mode_8bit: bool,
    device: torch.device,
) -> dict[int, dict[str, np.ndarray | float]]:
    normalizer = ai8x.normalize(SimpleNamespace(act_mode_8bit=act_mode_8bit))
    results: dict[int, dict[str, np.ndarray | float]] = {}
    num_samples = int(targets.numel())

    for prefix in prefixes:
        all_preds = np.empty(num_samples, dtype=np.int64)
        all_conf = np.empty(num_samples, dtype=np.float32)
        all_margin = np.empty(num_samples, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                inputs = data[start:end].float().div(255.0)
                inputs = normalizer(inputs)
                if prefix < inputs.shape[1]:
                    inputs[:, prefix:, :, :] = 0
                inputs = inputs.to(device)

                logits = model(inputs)
                if act_mode_8bit:
                    logits = logits / 128.0
                probs = torch.softmax(logits, dim=1)
                top2 = torch.topk(probs, k=2, dim=1)
                preds = top2.indices[:, 0].cpu().numpy()
                conf = top2.values[:, 0].cpu().numpy()
                margin = (top2.values[:, 0] - top2.values[:, 1]).cpu().numpy()

                all_preds[start:end] = preds
                all_conf[start:end] = conf
                all_margin[start:end] = margin

        truth = targets.cpu().numpy()
        accuracy = float(np.mean(all_preds == truth))
        results[prefix] = {
            "preds": all_preds,
            "confidence": all_conf,
            "margin": all_margin,
            "accuracy": accuracy,
            "mean_confidence": float(np.mean(all_conf)),
        }
        print(
            f"prefix={prefix:2d} accuracy={accuracy:.6f} "
            f"mean_conf={float(np.mean(all_conf)):.6f}",
            flush=True,
        )

    return results


def sweep_policies(
    results: dict[int, dict[str, np.ndarray | float]],
    targets: torch.Tensor,
    prefixes: list[int],
    thresholds: list[float],
    margins: list[float],
) -> list[dict[str, object]]:
    truth = targets.cpu().numpy()
    full_prefix = max(prefixes)
    full_accuracy = float(results[full_prefix]["accuracy"])
    rows: list[dict[str, object]] = []

    for threshold in thresholds:
        for margin_threshold in margins:
            accepted_prefix = np.full_like(truth, full_prefix, dtype=np.int64)
            accepted_pred = np.asarray(results[full_prefix]["preds"]).copy()
            accepted_conf = np.asarray(results[full_prefix]["confidence"]).copy()
            accepted_margin = np.asarray(results[full_prefix]["margin"]).copy()
            undecided = np.ones_like(truth, dtype=bool)

            for prefix in prefixes:
                preds = np.asarray(results[prefix]["preds"])
                conf = np.asarray(results[prefix]["confidence"])
                margin = np.asarray(results[prefix]["margin"])
                accept = undecided & (conf >= threshold) & (margin >= margin_threshold)
                if np.any(accept):
                    accepted_prefix[accept] = prefix
                    accepted_pred[accept] = preds[accept]
                    accepted_conf[accept] = conf[accept]
                    accepted_margin[accept] = margin[accept]
                    undecided[accept] = False

            accuracy = float(np.mean(accepted_pred == truth))
            avg_prefix = float(np.mean(accepted_prefix))
            early_rate = float(np.mean(accepted_prefix < full_prefix))
            saving = float(1.0 - avg_prefix / full_prefix)
            rows.append(
                {
                    "threshold": f"{threshold:.2f}",
                    "margin": f"{margin_threshold:.2f}",
                    "accuracy": f"{accuracy:.6f}",
                    "accuracy_drop_vs_full": f"{full_accuracy - accuracy:.6f}",
                    "avg_prefix": f"{avg_prefix:.4f}",
                    "cnn_compute_saving": f"{saving:.6f}",
                    "early_accept_rate": f"{early_rate:.6f}",
                    "mean_accept_confidence": f"{float(np.mean(accepted_conf)):.6f}",
                    "mean_accept_margin": f"{float(np.mean(accepted_margin)):.6f}",
                }
            )

    return rows


def train_class_thresholds(
    results: dict[int, dict[str, np.ndarray | float]],
    truth: np.ndarray,
    tune_mask: np.ndarray,
    prefixes: list[int],
    thresholds: list[float],
    min_precision: float,
    num_classes: int,
) -> dict[int, np.ndarray]:
    table: dict[int, np.ndarray] = {}
    for prefix in prefixes[:-1]:
        pred = np.asarray(results[prefix]["preds"])
        conf = np.asarray(results[prefix]["confidence"])
        prefix_thresholds = np.full(num_classes, 1.01, dtype=np.float32)
        for class_idx in range(num_classes):
            class_mask = tune_mask & (pred == class_idx)
            if not np.any(class_mask):
                continue
            chosen = 1.01
            best_count = -1
            for threshold in thresholds:
                accept = class_mask & (conf >= threshold)
                count = int(np.sum(accept))
                if count == 0:
                    continue
                precision = float(np.mean(pred[accept] == truth[accept]))
                if precision >= min_precision and count > best_count:
                    chosen = threshold
                    best_count = count
            prefix_thresholds[class_idx] = chosen
        table[prefix] = prefix_thresholds
    return table


def eval_class_thresholds(
    results: dict[int, dict[str, np.ndarray | float]],
    truth: np.ndarray,
    eval_mask: np.ndarray,
    prefixes: list[int],
    table: dict[int, np.ndarray],
) -> dict[str, float]:
    full_prefix = max(prefixes)
    accepted_prefix = np.full_like(truth, full_prefix, dtype=np.int64)
    accepted_pred = np.asarray(results[full_prefix]["preds"]).copy()
    undecided = eval_mask.copy()

    for prefix in prefixes[:-1]:
        pred = np.asarray(results[prefix]["preds"])
        conf = np.asarray(results[prefix]["confidence"])
        thresholds = table[prefix]
        accept = undecided & (conf >= thresholds[pred])
        if np.any(accept):
            accepted_prefix[accept] = prefix
            accepted_pred[accept] = pred[accept]
            undecided[accept] = False

    eval_truth = truth[eval_mask]
    eval_pred = accepted_pred[eval_mask]
    eval_prefix = accepted_prefix[eval_mask]
    return {
        "accuracy": float(np.mean(eval_pred == eval_truth)),
        "avg_prefix": float(np.mean(eval_prefix)),
        "cnn_compute_saving": float(1.0 - np.mean(eval_prefix) / full_prefix),
        "early_accept_rate": float(np.mean(eval_prefix < full_prefix)),
    }


def threshold_table_to_jsonable(table: dict[int, np.ndarray], labels: list[str]) -> dict[str, object]:
    return {
        str(prefix): [
            {
                "class_id": class_idx,
                "label": labels[class_idx],
                "threshold": None if float(threshold) > 1.0 else round(float(threshold), 4),
            }
            for class_idx, threshold in enumerate(thresholds)
        ]
        for prefix, thresholds in table.items()
    }


def sweep_class_specific_policies(
    results: dict[int, dict[str, np.ndarray | float]],
    targets: torch.Tensor,
    prefixes: list[int],
    thresholds: list[float],
    min_precisions: list[float],
    num_classes: int,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    truth = targets.cpu().numpy()
    tune_mask = np.zeros_like(truth, dtype=bool)
    tune_mask[::2] = True
    eval_mask = ~tune_mask
    full_prefix = max(prefixes)
    full_preds = np.asarray(results[full_prefix]["preds"])
    eval_full_accuracy = float(np.mean(full_preds[eval_mask] == truth[eval_mask]))

    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for min_precision in min_precisions:
        table = train_class_thresholds(
            results,
            truth,
            tune_mask,
            prefixes,
            thresholds,
            min_precision,
            num_classes,
        )
        metrics = eval_class_thresholds(results, truth, eval_mask, prefixes, table)
        row = {
            "min_precision": f"{min_precision:.2f}",
            "eval_full_accuracy": f"{eval_full_accuracy:.6f}",
            "eval_accuracy": f"{metrics['accuracy']:.6f}",
            "accuracy_drop_vs_eval_full": f"{eval_full_accuracy - metrics['accuracy']:.6f}",
            "avg_prefix": f"{metrics['avg_prefix']:.4f}",
            "cnn_compute_saving": f"{metrics['cnn_compute_saving']:.6f}",
            "early_accept_rate": f"{metrics['early_accept_rate']:.6f}",
        }
        rows.append(row)
        is_valid = float(row["accuracy_drop_vs_eval_full"]) <= 0.01
        if is_valid and (
            best is None or float(row["cnn_compute_saving"]) > float(best["cnn_compute_saving"])
        ):
            best = row

    return rows, best


def main() -> int:
    args = parse_args()
    prefixes = sorted(parse_int_list(args.prefixes))
    thresholds = parse_float_list(args.thresholds)
    margins = parse_float_list(args.margins)
    class_thresholds = parse_float_list(args.class_threshold_grid)
    class_min_precisions = parse_float_list(args.class_min_precision)

    _, labels = load_state_dict(args.checkpoint)
    model, ai8x = load_model(args, labels)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    model.eval()

    payload = torch.load(args.cache / "val.pt", map_location="cpu")
    data = payload["data"]
    targets = payload["labels"].long()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    prefix_results = evaluate_prefixes(
        model,
        ai8x,
        data,
        targets,
        labels,
        prefixes,
        args.batch_size,
        args.act_mode_8bit,
        device,
    )

    prefix_rows = [
        {
            "prefix": prefix,
            "accuracy": f"{float(prefix_results[prefix]['accuracy']):.6f}",
            "mean_confidence": f"{float(prefix_results[prefix]['mean_confidence']):.6f}",
        }
        for prefix in prefixes
    ]
    write_csv(
        args.output_dir / "prefix_accuracy.csv",
        prefix_rows,
        ["prefix", "accuracy", "mean_confidence"],
    )

    policy_rows = sweep_policies(prefix_results, targets, prefixes, thresholds, margins)
    write_csv(
        args.output_dir / "early_exit_policy_sweep.csv",
        policy_rows,
        [
            "threshold",
            "margin",
            "accuracy",
            "accuracy_drop_vs_full",
            "avg_prefix",
            "cnn_compute_saving",
            "early_accept_rate",
            "mean_accept_confidence",
            "mean_accept_margin",
        ],
    )

    full_accuracy = float(prefix_results[max(prefixes)]["accuracy"])
    candidates = [
        row
        for row in policy_rows
        if float(row["accuracy_drop_vs_full"]) <= args.max_accuracy_drop
    ]
    candidates.sort(key=lambda row: (-float(row["cnn_compute_saving"]), -float(row["accuracy"])))
    top_candidates = candidates[:10]
    write_csv(
        args.output_dir / "recommended_policies.csv",
        top_candidates,
        [
            "threshold",
            "margin",
            "accuracy",
            "accuracy_drop_vs_full",
            "avg_prefix",
            "cnn_compute_saving",
            "early_accept_rate",
            "mean_accept_confidence",
            "mean_accept_margin",
        ],
    )

    summary = {
        "samples": int(targets.numel()),
        "prefixes": prefixes,
        "full_prefix": max(prefixes),
        "full_accuracy": full_accuracy,
        "max_accuracy_drop": args.max_accuracy_drop,
        "best_recommended": top_candidates[0] if top_candidates else None,
    }

    class_policy_rows, best_class_policy = sweep_class_specific_policies(
        prefix_results,
        targets,
        prefixes,
        class_thresholds,
        class_min_precisions,
        len(labels),
    )
    write_csv(
        args.output_dir / "class_specific_policy_sweep.csv",
        class_policy_rows,
        [
            "min_precision",
            "eval_full_accuracy",
            "eval_accuracy",
            "accuracy_drop_vs_eval_full",
            "avg_prefix",
            "cnn_compute_saving",
            "early_accept_rate",
        ],
    )
    summary["best_class_specific_policy"] = best_class_policy
    if best_class_policy is not None:
        best_min_precision = float(best_class_policy["min_precision"])
        full_mask = np.ones(int(targets.numel()), dtype=bool)
        full_table = train_class_thresholds(
            prefix_results,
            targets.cpu().numpy(),
            full_mask,
            prefixes,
            class_thresholds,
            best_min_precision,
            len(labels),
        )
        class_table_payload = {
            "selected_min_precision": best_min_precision,
            "note": "Thresholds were refit on the full validation set for deployment convenience.",
            "prefix_thresholds": threshold_table_to_jsonable(full_table, labels),
        }
        (args.output_dir / "class_thresholds_best_fullval.json").write_text(
            json.dumps(class_table_payload, indent=2),
            encoding="utf-8",
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest recommended policies:")
    for row in top_candidates[:5]:
        print(
            "threshold={threshold} margin={margin} accuracy={accuracy} "
            "drop={accuracy_drop_vs_full} avg_prefix={avg_prefix} "
            "saving={cnn_compute_saving} early_rate={early_accept_rate}".format(**row)
        )

    print("\nClass-specific policy sweep:")
    for row in class_policy_rows:
        print(
            "min_precision={min_precision} eval_accuracy={eval_accuracy} "
            "drop={accuracy_drop_vs_eval_full} avg_prefix={avg_prefix} "
            "saving={cnn_compute_saving} early_rate={early_accept_rate}".format(**row)
        )

    print(f"\noutput_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
