#!/usr/bin/env python3
"""Simulate a sliding-window early-stop state machine for Jester.

The firmware still updates the 12-frame sliding history at the camera rate. On
every non-held frame, it immediately probes prefix 8, then prefix 10, then full
12 only if needed.  A confident prefix-8 decision suppresses the next 4 frame
CNN calls, and a confident prefix-10 decision suppresses the next 2 frame CNN
calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai8x-training", type=Path, default=Path("ai8x-training"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/ai8x_jestermotion12_20260520_184836/"
            "qat_artifacts/ai8x_jestermotion12_qat_20260520_184836_qat_best.pth.tar"
        ),
    )
    parser.add_argument("--cache", type=Path, default=Path("cache/jester_cache_motion12_64_full"))
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(
            "runs/early_exit_jestermotion12_deploy_safe90_20260523/"
            "class_thresholds_best_fullval.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sliding_earlystop_state_machine_20260530"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--sequence-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=78000)
    parser.add_argument("--act-mode-8bit", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def import_eval_helpers(project_root: Path):
    sys.path.insert(0, str(project_root))
    from eval_jester_early_exit import (  # pylint: disable=import-outside-toplevel
        evaluate_prefixes,
        load_model,
        load_state_dict,
    )

    return evaluate_prefixes, load_model, load_state_dict


def load_threshold_table(path: Path, prefixes: tuple[int, ...], num_classes: int) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["prefix_thresholds"]
    table: dict[int, np.ndarray] = {}
    for prefix in prefixes:
        values = np.full(num_classes, 1.01, dtype=np.float32)
        for item in raw[str(prefix)]:
            threshold = item.get("threshold")
            if threshold is not None:
                values[int(item["class_id"])] = float(threshold)
        table[prefix] = values
    return table


def state_machine_for_indices(
    indices: np.ndarray,
    truth: np.ndarray,
    prefix_results: dict[int, dict[str, np.ndarray | float]],
    threshold_table: dict[int, np.ndarray],
) -> dict[str, float | int]:
    prefix8_pred = np.asarray(prefix_results[8]["preds"])
    prefix8_conf = np.asarray(prefix_results[8]["confidence"])
    prefix10_pred = np.asarray(prefix_results[10]["preds"])
    prefix10_conf = np.asarray(prefix_results[10]["confidence"])
    full_pred = np.asarray(prefix_results[12]["preds"])

    fresh_correct = 0
    held_correct = 0
    calls = 0
    fresh_decisions = 0
    held_frames = 0
    hold_cooldown = 0
    held_decision = -1
    prefix_counts = {8: 0, 10: 0, 12: 0}

    for idx in indices:
        if hold_cooldown > 0:
            hold_cooldown -= 1
            held_frames += 1
            held_correct += int(held_decision == int(truth[idx]))
            continue

        calls += 1
        pred8 = int(prefix8_pred[idx])
        if float(prefix8_conf[idx]) >= float(threshold_table[8][pred8]):
            decision = pred8
            hold_cooldown = 4
            prefix_counts[8] += 1
        else:
            calls += 1
            pred10 = int(prefix10_pred[idx])
            if float(prefix10_conf[idx]) >= float(threshold_table[10][pred10]):
                decision = pred10
                hold_cooldown = 2
                prefix_counts[10] += 1
            else:
                calls += 1
                decision = int(full_pred[idx])
                prefix_counts[12] += 1

        held_decision = decision
        fresh_decisions += 1
        fresh_correct += int(decision == int(truth[idx]))
        held_correct += int(decision == int(truth[idx]))

    frames = int(len(indices))
    full_sliding_calls = frames
    always_probe_calls = frames * 3
    avg_calls = calls / frames
    active_frames = fresh_decisions
    return {
        "frames": frames,
        "fresh_decision_accuracy": fresh_correct / max(active_frames, 1),
        "stream_accuracy_with_hold": held_correct / frames,
        "cnn_calls": calls,
        "avg_cnn_calls_per_frame": avg_calls,
        "effective_cnn_fps_at_12fps_camera": avg_calls * 12.0,
        "saving_vs_full12_every_frame": 1.0 - calls / full_sliding_calls,
        "saving_vs_always_probe_every_frame": 1.0 - calls / always_probe_calls,
        "active_frames": active_frames,
        "held_frames": held_frames,
        "held_frame_rate": held_frames / frames,
        "accepted8": prefix_counts[8],
        "accepted10": prefix_counts[10],
        "full12": prefix_counts[12],
        "accepted8_rate": prefix_counts[8] / max(active_frames, 1),
        "accepted10_rate": prefix_counts[10] / max(active_frames, 1),
        "full12_rate": prefix_counts[12] / max(active_frames, 1),
    }


def summarize_trials(rows: list[dict[str, float | int]]) -> dict[str, float]:
    fields = (
        "fresh_decision_accuracy",
        "stream_accuracy_with_hold",
        "avg_cnn_calls_per_frame",
        "effective_cnn_fps_at_12fps_camera",
        "saving_vs_full12_every_frame",
        "saving_vs_always_probe_every_frame",
        "held_frame_rate",
        "accepted8_rate",
        "accepted10_rate",
        "full12_rate",
    )
    summary: dict[str, float] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        summary[f"{field}_mean"] = float(np.mean(values))
        summary[f"{field}_std"] = float(np.std(values))
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    evaluate_prefixes, load_model, load_state_dict = import_eval_helpers(project_root)

    _, labels = load_state_dict(args.checkpoint)
    model_args = SimpleNamespace(
        ai8x_training=args.ai8x_training,
        checkpoint=args.checkpoint,
        model="jestermotion12",
        act_mode_8bit=args.act_mode_8bit,
    )
    model, ai8x = load_model(model_args, labels)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    model.eval()

    payload = torch.load(args.cache / "val.pt", map_location="cpu")
    data = payload["data"]
    targets = payload["labels"].long()
    truth = targets.cpu().numpy()
    prefixes = [8, 10, 12]

    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    threshold_table = load_threshold_table(args.thresholds, (8, 10), len(labels))
    all_indices = np.arange(int(targets.numel()))
    full_validation = state_machine_for_indices(all_indices, truth, prefix_results, threshold_table)

    rng = np.random.default_rng(args.seed)
    sequence_size = int(args.sequence_size)
    if sequence_size <= 0 or sequence_size > len(all_indices):
        sequence_size = len(all_indices)

    trial_rows: list[dict[str, float | int]] = []
    for trial in range(args.trials):
        replace = sequence_size > len(all_indices)
        indices = rng.choice(all_indices, size=sequence_size, replace=replace)
        row = state_machine_for_indices(indices, truth, prefix_results, threshold_table)
        row["trial"] = trial
        row["seed"] = args.seed
        trial_rows.append(row)

    prefix_accuracy = {
        str(prefix): {
            "accuracy": float(prefix_results[prefix]["accuracy"]),
            "mean_confidence": float(prefix_results[prefix]["mean_confidence"]),
        }
        for prefix in prefixes
    }
    summary = {
        "note": "Kaggle test labels are not available locally, so this uses the labeled validation cache.",
        "state_machine": "sliding stream: every non-held frame probes prefix8; if accepted, hold/skip 4 frames; else probes prefix10; if accepted, hold/skip 2 frames; else probes prefix12/full.",
        "samples": int(targets.numel()),
        "trial_sequence_size": sequence_size,
        "trials": args.trials,
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "thresholds": str(args.thresholds),
        "prefix_accuracy": prefix_accuracy,
        "full_validation": full_validation,
        "random_trial_summary": summarize_trials(trial_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "random_trials.csv", trial_rows)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
