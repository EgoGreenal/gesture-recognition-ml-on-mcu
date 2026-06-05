#!/usr/bin/env python3
"""Collect Jester training histories, summary metrics, durations, params, and MMAC.

Run this on the remote training server from the 20bn-jester project root.
The output JSON is consumed by the local Word report builder.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT = Path("/path/to/20bn-jester")
RUNS = PROJECT / "runs"
AI8X = PROJECT / "ai8x-training"


@dataclass(frozen=True)
class RunSpec:
    run_dir: str
    display: str
    arch: str
    input_desc: str
    deployable: bool = True
    notes: str = ""
    include_distill: bool = False
    include_qat_sweep: bool = False


RUN_SPECS = [
    RunSpec(
        "cached_tinygesture_full_baseline_20260519_192917",
        "TinyGesture raw 8-frame software baseline",
        "tinygesture",
        "8 raw grayscale frames, 64x64",
        deployable=False,
        notes="PyTorch software baseline before ai8x conversion.",
    ),
    RunSpec(
        "cached_tinygesture_full_finetune300_20260519_201404",
        "TinyGesture raw 8-frame 300-epoch finetune",
        "tinygesture",
        "8 raw grayscale frames, 64x64",
        deployable=False,
        notes="Longer software-only baseline run.",
    ),
    RunSpec(
        "ai8x_jesternet_qat_20260520_005821",
        "JesterNet raw 8-frame ai8x QAT",
        "jesternet",
        "8 raw grayscale frames, 64x64",
    ),
    RunSpec(
        "ai8x_jester12wide_20260520_161242",
        "Jester12Wide raw 12-frame",
        "jester12wide",
        "12 raw grayscale frames, 64x64",
    ),
    RunSpec(
        "ai8x_jestermotion8_20260520_211203",
        "JesterMotion8 motion-channel ablation",
        "jestermotion8",
        "anchor + 7 absolute-motion planes, 64x64",
    ),
    RunSpec(
        "ai8x_jestermotion12_20260520_184836",
        "JesterMotion12 motion-channel baseline",
        "jestermotion12",
        "anchor + 11 absolute-motion planes, 64x64",
    ),
    RunSpec(
        "ai8x_jestermotion12_qat_lowlr_20260521_141542",
        "JesterMotion12 low-LR QAT",
        "jestermotion12",
        "anchor + 11 absolute-motion planes, 64x64",
        notes="QAT-only LR sweep around the motion12 baseline.",
    ),
    RunSpec(
        "ai8x_jestermixed12_20260521_141543",
        "JesterMixed12 raw/motion mix",
        "jestermixed12",
        "4 raw planes + 8 absolute-motion planes, 64x64",
    ),
    RunSpec(
        "ai8x_jestersignedmotion12_20260521_174618",
        "JesterSignedMotion12",
        "jestersignedmotion12",
        "anchor + 11 signed-motion planes, 64x64",
    ),
    RunSpec(
        "ai8x_jestermotion12_hard_20260521_192437",
        "JesterMotion12 hard-class oversampling",
        "jestermotion12",
        "anchor + 11 absolute-motion planes, 64x64",
        notes="Training-set reweighting/oversampling variant.",
    ),
    RunSpec(
        "ai8x_jestermotion12res_20260521_163044",
        "JesterMotion12 residual",
        "jestermotion12res",
        "anchor + 11 absolute-motion planes, 64x64",
    ),
    RunSpec(
        "ai8x_jestermotion12res_signed_20260521_225051",
        "JesterMotion12 residual + signed motion",
        "jestermotion12res",
        "anchor + 11 signed-motion planes, 64x64",
    ),
    RunSpec(
        "ai8x_jestermotion12resfc192_20260522_000610",
        "JesterMotion12 residual fc192",
        "jestermotion12resfc192",
        "anchor + 11 absolute-motion planes, 64x64",
    ),
    RunSpec(
        "ai8x_jestermotion12resfc192_signed_20260522_101448",
        "JesterMotion12 residual fc192 + signed motion",
        "jestermotion12resfc192",
        "anchor + 11 signed-motion planes, 64x64",
    ),
    RunSpec(
        "ai8x_jestermotion12resxconv_20260522_012825",
        "JesterMotion12 residual + extra conv",
        "jestermotion12resxconv",
        "anchor + 11 absolute-motion planes, 64x64",
    ),
    RunSpec(
        "ai8x_jestermotion12resxconv_signed_20260522_113043",
        "JesterMotion12 residual + extra conv + signed motion",
        "jestermotion12resxconv",
        "anchor + 11 signed-motion planes, 64x64",
    ),
    RunSpec(
        "ai8x_jestermotion12resxconv_signed_distill_20260522_133850",
        "Big-teacher distilled signed resxconv student",
        "jestermotion12resxconv",
        "anchor + 11 signed-motion planes, 64x64",
        notes="Includes software teacher, distillation stage, and ai8x QAT.",
        include_distill=True,
    ),
    RunSpec(
        "ai8x_jestermotion12resxconv_signed_huge_teacher_distill_20260526_155122",
        "Huge-teacher distilled signed resxconv student",
        "jestermotion12resxconv",
        "anchor + 11 signed-motion planes, 64x64",
        notes="Includes 11.4M-param software teacher, distillation stage, and ai8x QAT.",
        include_distill=True,
    ),
    RunSpec(
        "ai8x_jestermotion12resxconv_signed_huge_teacher_qat_sweep_20260528_102149",
        "Huge-teacher student QAT LR/epoch sweep",
        "jestermotion12resxconv",
        "anchor + 11 signed-motion planes, 64x64",
        notes="Four QAT schedules launched from the huge-teacher distilled student.",
        include_qat_sweep=True,
    ),
]


TIMESTAMP_RE = re.compile(r"^\[(?P<stamp>[A-Z][a-z]{2} [A-Z][a-z]{2} +\d+ \d\d:\d\d:\d\d CEST 2026)\] (?P<msg>.*)")
AI8X_TRAIN_RE = re.compile(
    r"Epoch: \[(?P<epoch>\d+)\]\[\s*(?P<step>\d+)/\s*(?P<steps>\d+)\]\s+"
    r"Overall Loss (?P<loss>[-+0-9.eE]+).*?"
    r"(?:Top1 (?P<top1>[-+0-9.eE]+)\s+Top5 (?P<top5>[-+0-9.eE]+).*?)?"
    r"LR (?P<lr>[-+0-9.eE]+)\s+Time (?P<time>[-+0-9.eE]+)"
)
AI8X_VAL_HEADER_RE = re.compile(r"--- validate \(epoch=(?P<epoch>\d+)\)")
AI8X_RESULT_RE = re.compile(
    r"==> Top1: (?P<top1>[-+0-9.eE]+)\s+Top5: (?P<top5>[-+0-9.eE]+)\s+Loss: (?P<loss>[-+0-9.eE]+)"
)
AI8X_BEST_RE = re.compile(
    r"==> Best \[Top1: (?P<top1>[-+0-9.eE]+)\s+Top5: (?P<top5>[-+0-9.eE]+)\s+Params: (?P<params>\d+) on epoch: (?P<epoch>\d+)\]"
)
DISTILL_STEP_RE = re.compile(
    r"(?P<phase>teacher|student) epoch=(?P<epoch>\d+) step=(?P<step>\d+)/(?P<steps>\d+) "
    r"loss=(?P<loss>[-+0-9.eE]+).*?top1=(?P<top1>[-+0-9.eE]+) lr=(?P<lr>[-+0-9.eE]+)"
)
DISTILL_SUMMARY_RE = re.compile(
    r"(?P<tag>TEACHER|STUDENT) epoch=(?P<epoch>\d+) val_top1=(?P<top1>[-+0-9.eE]+) "
    r"val_top5=(?P<top5>[-+0-9.eE]+) val_loss=(?P<loss>[-+0-9.eE]+) "
    r"best_top1=(?P<best>[-+0-9.eE]+) time=(?P<time>[-+0-9.eE]+)s"
)


def parse_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp.replace("  ", " "), "%a %b %d %H:%M:%S CEST %Y")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_duration_from_train_log(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    events: list[tuple[datetime, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TIMESTAMP_RE.match(line)
        if m:
            events.append((parse_stamp(m.group("stamp")), m.group("msg")))
    durations: dict[str, float] = {}
    for idx, (stamp, msg) in enumerate(events):
        next_stamp = events[idx + 1][0] if idx + 1 < len(events) else None
        if "floating-point pretrain" in msg and next_stamp:
            durations["fp"] = (next_stamp - stamp).total_seconds()
        elif "QAT fine-tune" in msg and next_stamp:
            durations["qat"] = (next_stamp - stamp).total_seconds()
        elif "starting ai8x JesterNet QAT" in msg and next_stamp:
            durations["qat"] = (next_stamp - stamp).total_seconds()
        elif "QAT sweep start" in msg and next_stamp:
            suffix = re.search(r"suffix=([^ ]+)", msg)
            if suffix:
                durations[f"qat_{suffix.group(1)}"] = (next_stamp - stamp).total_seconds()
    return durations


def parse_ai8x_log(path: Path, phase_name: str) -> dict[str, Any]:
    epochs: dict[int, dict[str, Any]] = {}
    best: dict[str, Any] = {}
    current_val_epoch: int | None = None
    in_test_best = False
    test_best: dict[str, float] | None = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        mt = AI8X_TRAIN_RE.search(line)
        if mt and int(mt.group("step")) == int(mt.group("steps")):
            epoch = int(mt.group("epoch"))
            row = epochs.setdefault(epoch, {"epoch": epoch})
            row.update(
                {
                    "train_loss": float(mt.group("loss")),
                    "train_top1": float(mt.group("top1")) if mt.group("top1") else None,
                    "train_top5": float(mt.group("top5")) if mt.group("top5") else None,
                    "lr": float(mt.group("lr")),
                    "batch_time_s": float(mt.group("time")),
                    "train_steps": int(mt.group("steps")),
                }
            )
            continue

        mh = AI8X_VAL_HEADER_RE.search(line)
        if mh:
            current_val_epoch = int(mh.group("epoch"))
            in_test_best = False
            continue

        if line.startswith("--- test (best)"):
            current_val_epoch = None
            in_test_best = True
            continue
        if line.startswith("--- test"):
            current_val_epoch = None
            in_test_best = False
            continue

        mr = AI8X_RESULT_RE.search(line)
        if mr:
            row_data = {
                "val_top1": float(mr.group("top1")),
                "val_top5": float(mr.group("top5")),
                "val_loss": float(mr.group("loss")),
            }
            if current_val_epoch is not None:
                row = epochs.setdefault(current_val_epoch, {"epoch": current_val_epoch})
                row.update(row_data)
            elif in_test_best:
                test_best = {
                    "top1": row_data["val_top1"],
                    "top5": row_data["val_top5"],
                    "loss": row_data["val_loss"],
                }
            continue

        mb = AI8X_BEST_RE.search(line)
        if mb:
            best = {
                "top1": float(mb.group("top1")),
                "top5": float(mb.group("top5")),
                "params": int(mb.group("params")),
                "epoch": int(mb.group("epoch")),
            }

    curve = [epochs[key] for key in sorted(epochs)]
    if not best and curve:
        valid = [row for row in curve if row.get("val_top1") is not None]
        if valid:
            row = max(valid, key=lambda r: r["val_top1"])
            best = {
                "top1": row["val_top1"],
                "top5": row.get("val_top5"),
                "params": None,
                "epoch": row["epoch"],
            }

    estimated_duration_s = sum(
        float(row.get("batch_time_s") or 0.0) * int(row.get("train_steps") or 0)
        for row in curve
    ) or None

    return {
        "name": phase_name,
        "source": str(path.relative_to(PROJECT)),
        "epochs": curve,
        "best": best,
        "test_best": test_best,
        "estimated_train_duration_s": estimated_duration_s,
    }


def parse_history_json(path: Path) -> dict[str, Any]:
    rows = read_json(path)
    curve = []
    for row in rows:
        curve.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_top1": float(row["train_acc"]) * 100.0,
                "val_loss": float(row["val_loss"]),
                "val_top1": float(row["val_acc"]) * 100.0,
                "val_top5": None,
                "lr": float(row.get("lr", 0.0)),
            }
        )
    best_row = max(curve, key=lambda r: r["val_top1"]) if curve else {}
    return {
        "name": "software_train",
        "source": str(path.relative_to(PROJECT)),
        "epochs": curve,
        "best": {
            "top1": best_row.get("val_top1"),
            "top5": None,
            "params": None,
            "epoch": best_row.get("epoch"),
        } if best_row else {},
    }


def parse_distill_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    phases: dict[str, dict[int, dict[str, Any]]] = {"teacher": {}, "student": {}}
    seconds: dict[str, float] = {"teacher": 0.0, "student": 0.0}
    best: dict[str, dict[str, Any]] = {"teacher": {}, "student": {}}

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ms = DISTILL_STEP_RE.search(line)
        if ms and int(ms.group("step")) == int(ms.group("steps")):
            phase = ms.group("phase")
            epoch = int(ms.group("epoch"))
            phases[phase].setdefault(epoch, {"epoch": epoch}).update(
                {
                    "train_loss": float(ms.group("loss")),
                    "train_top1": float(ms.group("top1")),
                    "lr": float(ms.group("lr")),
                }
            )
            continue
        mt = DISTILL_SUMMARY_RE.search(line)
        if mt:
            phase = "teacher" if mt.group("tag") == "TEACHER" else "student"
            epoch = int(mt.group("epoch"))
            row = phases[phase].setdefault(epoch, {"epoch": epoch})
            row.update(
                {
                    "val_loss": float(mt.group("loss")),
                    "val_top1": float(mt.group("top1")),
                    "val_top5": float(mt.group("top5")),
                }
            )
            seconds[phase] += float(mt.group("time"))
            if not best[phase] or row["val_top1"] > best[phase]["top1"]:
                best[phase] = {
                    "top1": row["val_top1"],
                    "top5": row["val_top5"],
                    "epoch": epoch,
                }

    result = []
    for phase in ("teacher", "student"):
        curve = [phases[phase][key] for key in sorted(phases[phase])]
        if curve:
            result.append(
                {
                    "name": f"distill_{phase}",
                    "source": str(path.relative_to(PROJECT)),
                    "epochs": curve,
                    "best": best[phase],
                    "duration_s": seconds[phase] or None,
                }
            )
    return result


def find_single_log(run_path: Path, subdir: str) -> Path | None:
    folder = run_path / subdir
    logs = sorted(folder.glob("*.log"))
    return logs[-1] if logs else None


def collect_phases(spec: RunSpec) -> tuple[list[dict[str, Any]], dict[str, float]]:
    run_path = RUNS / spec.run_dir
    phases: list[dict[str, Any]] = []
    durations = phase_duration_from_train_log(run_path / "train.log")

    if (run_path / "history.json").is_file():
        phases.append(parse_history_json(run_path / "history.json"))

    if spec.include_distill:
        phases.extend(parse_distill_log(run_path / "train.log"))

    if spec.include_qat_sweep:
        for log in sorted(run_path.glob("**/*.log")):
            if "train.log" in str(log):
                continue
            suffix = "qat"
            m = re.search(r"_(lr[^_/]+_ep\d+)_", log.name)
            if m:
                suffix = f"qat_{m.group(1)}"
            phase = parse_ai8x_log(log, suffix)
            if phase["epochs"]:
                phase["duration_s"] = durations.get(suffix) or phase.get("estimated_train_duration_s")
                phases.append(phase)
        # Fallback: parse the monolithic train.log if artifact logs were not copied.
        if not phases:
            phase = parse_ai8x_log(run_path / "train.log", "qat_sweep_all")
            if phase["epochs"]:
                phases.append(phase)
        return phases, durations

    for subdir, name in (("fp_artifacts", "fp"), ("qat_artifacts", "qat"), ("artifacts", "qat")):
        log = find_single_log(run_path, subdir)
        if log:
            phase = parse_ai8x_log(log, name)
            if phase["epochs"]:
                phase["duration_s"] = durations.get(name) or phase.get("estimated_train_duration_s")
                phases.append(phase)

    return phases, durations


def input_channels_for_arch(arch: str) -> int:
    return 8 if arch in {"tinygesture", "jesternet", "jestermotion8"} else 12


def import_model_for_arch(arch: str) -> nn.Module | None:
    sys.path.insert(0, str(AI8X))
    if arch == "tinygesture":
        mod = importlib.import_module("train_jester_cached")
        return mod.TinyGestureNet(input_frames=8, num_classes=27)

    import ai8x  # pylint: disable=import-error,import-outside-toplevel

    ai8x.set_device(85, False, False, verbose=False)
    module = importlib.import_module(f"models.{arch}")
    factory = getattr(module, arch)
    return factory(num_classes=27, num_channels=input_channels_for_arch(arch), dimensions=(64, 64), bias=True)


def build_teacher(variant: str) -> nn.Module | None:
    sys.path.insert(0, str(PROJECT))
    mod = importlib.import_module("train_jester_distill")
    config = mod.TEACHER_VARIANTS[variant]
    return mod.ConfigurableJesterTeacher(num_classes=27, **config)


def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, Any]:
    macs = 0
    layer_rows: list[dict[str, Any]] = []
    hooks = []

    def hook(name: str, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        layer_macs = 0
        op = getattr(module, "op", None)
        counted = op if isinstance(op, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)) else module
        if isinstance(counted, nn.Conv1d):
            out = output
            out_elements = int(out.shape[1] * out.shape[2])
            kernel_ops = int(counted.kernel_size[0] * counted.in_channels / counted.groups)
            layer_macs = out_elements * kernel_ops
        elif isinstance(counted, (nn.Conv2d, nn.ConvTranspose2d)):
            out = output
            out_elements = int(out.shape[1] * out.shape[2] * out.shape[3])
            kernel_ops = int(counted.kernel_size[0] * counted.kernel_size[1] * counted.in_channels / counted.groups)
            layer_macs = out_elements * kernel_ops
        elif isinstance(counted, nn.Conv3d):
            out = output
            out_elements = int(out.shape[1] * out.shape[2] * out.shape[3] * out.shape[4])
            kernel_ops = int(counted.kernel_size[0] * counted.kernel_size[1] * counted.kernel_size[2] * counted.in_channels / counted.groups)
            layer_macs = out_elements * kernel_ops
        elif isinstance(counted, nn.Linear):
            layer_macs = int(counted.in_features * counted.out_features)
        if layer_macs:
            macs += layer_macs
            layer_rows.append({"name": name, "type": counted.__class__.__name__, "macs": layer_macs})

    for name, module in model.named_modules():
        op = getattr(module, "op", None)
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)) \
                or isinstance(op, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)):
            hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook(n, m, i, o)))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for h in hooks:
        h.remove()
    params = sum(p.numel() for p in model.parameters())
    return {"params": int(params), "macs": int(macs), "mmac": macs / 1_000_000.0, "layers": layer_rows}


def metric_profile_for_arch(arch: str) -> dict[str, Any]:
    try:
        model = import_model_for_arch(arch)
        if model is None:
            return {}
        return count_macs(model, (1, input_channels_for_arch(arch), 64, 64))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"error": repr(exc)}


def best_from_phases(phases: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for phase in phases:
        if phase.get("test_best"):
            candidates.append((phase["test_best"].get("top1"), phase["test_best"].get("top5"), phase["name"], "test_best"))
        if phase.get("best") and phase["best"].get("top1") is not None:
            candidates.append((phase["best"].get("top1"), phase["best"].get("top5"), phase["name"], "val_best"))
    candidates = [c for c in candidates if c[0] is not None and not math.isnan(float(c[0]))]
    if not candidates:
        return {}
    top1, top5, phase, source = max(candidates, key=lambda c: c[0])
    return {"top1": float(top1), "top5": None if top5 is None else float(top5), "phase": phase, "source": source}


def main() -> None:
    arch_profiles: dict[str, Any] = {}
    for arch in sorted({spec.arch for spec in RUN_SPECS}):
        arch_profiles[arch] = metric_profile_for_arch(arch)

    teacher_profiles = {}
    for variant in ("big", "huge"):
        try:
            teacher_profiles[variant] = count_macs(build_teacher(variant), (1, 12, 64, 64))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            teacher_profiles[variant] = {"error": repr(exc)}

    models = []
    for spec in RUN_SPECS:
        phases, durations = collect_phases(spec)
        profile = arch_profiles.get(spec.arch, {})
        params_from_phases = [
            phase.get("best", {}).get("params")
            for phase in phases
            if phase.get("best", {}).get("params")
        ]
        params = (params_from_phases[-1] if params_from_phases else None) or profile.get("params")
        phase_seconds = sum(float(phase.get("duration_s") or 0.0) for phase in phases)
        if not phase_seconds:
            phase_seconds = sum(float(v) for v in durations.values())
        best_phases = [p for p in phases if not (spec.include_distill and p.get("name") == "distill_teacher")]
        models.append(
            {
                "id": spec.run_dir,
                "display": spec.display,
                "arch": spec.arch,
                "input": spec.input_desc,
                "deployable": spec.deployable,
                "notes": spec.notes,
                "params": params,
                "mmac": profile.get("mmac"),
                "macs": profile.get("macs"),
                "phases": phases,
                "durations": durations,
                "training_time_s": phase_seconds or None,
                "best": best_from_phases(best_phases),
            }
        )

    # Add software teachers as separate rows so their parameter/MMAC scale is explicit.
    for run_id, variant, display in (
        ("ai8x_jestermotion12resxconv_signed_distill_20260522_133850", "big", "Big software teacher"),
        ("ai8x_jestermotion12resxconv_signed_huge_teacher_distill_20260526_155122", "huge", "Huge software teacher"),
    ):
        parent = next((m for m in models if m["id"] == run_id), None)
        teacher_phase = None
        if parent:
            teacher_phase = next((p for p in parent["phases"] if p["name"] == "distill_teacher"), None)
        profile = teacher_profiles.get(variant, {})
        models.append(
            {
                "id": f"{run_id}__teacher",
                "display": display,
                "arch": f"{variant}_teacher",
                "input": "12 signed-motion planes, 64x64",
                "deployable": False,
                "notes": "Software-only teacher used for distillation, not MAX78000 deployable.",
                "params": profile.get("params"),
                "mmac": profile.get("mmac"),
                "macs": profile.get("macs"),
                "phases": [teacher_phase] if teacher_phase else [],
                "durations": {},
                "training_time_s": teacher_phase.get("duration_s") if teacher_phase else None,
                "best": best_from_phases([teacher_phase]) if teacher_phase else {},
            }
        )

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": str(PROJECT),
        "excluded": ["Current true Conv3D experiment is excluded per user request."],
        "mmac_definition": "Conv/Linear multiply-accumulate count for one inference at 64x64 input; pooling, activation, add, and data movement are not included.",
        "arch_profiles": arch_profiles,
        "teacher_profiles": teacher_profiles,
        "models": models,
    }
    out_path = PROJECT / "runs" / "jester_training_report_data_20260530.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
