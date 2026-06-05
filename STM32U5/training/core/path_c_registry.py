"""Day 10 Path C: unified variant dispatch for train_student.py.

Resolves a variant name -> the right factory + KD mode + input mode.

Sequence (per-frame KD):
    V0..V9 (legacy tiny CNN), C1a-c (TSM-MBV2), C3a-c (hybrid)
Stacked (video-level KD):
    C2a-c (stacked MobileNet)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import lazily inside functions to avoid TF init when only spec is queried.


@dataclass
class VariantSpec:
    name: str
    family: str               # 'legacy' | 'C1' | 'C2' | 'C3'
    input_mode: str           # 'sequence' | 'stacked'
    kd_mode: str              # 'per_frame' | 'video_level'
    T: int = 8                # number of input frames
    img_size: int = 64
    in_channels: int = 1


# ---- C1 / C2 / C3 specs (param presets) -----------------------------------

C1_PRESETS = {
    "C1A": dict(width_mult=0.25, n_blocks=17, input_hw=64),
    "C1B": dict(width_mult=0.50, n_blocks=17, input_hw=64),
    "C1C": dict(width_mult=0.25, n_blocks=10, input_hw=64),
    # Day 10 v2 deep dive (C1b 84% — push to 85-90%):
    # Width sweep at full depth (17 blocks), 64x64:
    "C1D": dict(width_mult=0.35, n_blocks=17, input_hw=64),
    "C1E": dict(width_mult=0.40, n_blocks=17, input_hw=64),
    "C1F": dict(width_mult=0.75, n_blocks=17, input_hw=64),
    "C1G": dict(width_mult=1.00, n_blocks=17, input_hw=64),
    # Higher input resolution (w=0.5 same as C1b for ablation):
    "C1H": dict(width_mult=0.50, n_blocks=17, input_hw=96),
    "C1I": dict(width_mult=0.50, n_blocks=17, input_hw=112),
    # Same as C1b but longer training (epoch budget set by sbatch)
    "C1J": dict(width_mult=0.50, n_blocks=17, input_hw=64),
    # Day 10 v3 deployment-friendly sweet spot:
    # C1k: width 0.6 (between C1b 0.5 and C1f 0.75) at 64x64
    "C1K": dict(width_mult=0.60, n_blocks=17, input_hw=64),
    # C1m: w=0.5 at 80x80 (between 64 and 96 resolutions)
    "C1M": dict(width_mult=0.50, n_blocks=17, input_hw=80),
    # C1n: C1g full-width but truncated to 13 blocks (cut last stage to fit Flash)
    "C1N": dict(width_mult=1.00, n_blocks=13, input_hw=64),
    # C1o: w=0.8 17 blk — between C1f (w=0.75) and C1g (w=1.0), targeting <2MB Flash
    "C1O": dict(width_mult=0.80, n_blocks=17, input_hw=64),
}

# C2 sub-variant id is passed straight to build_c2
# C3 sub-variant id is passed straight to build_c3


def get_spec(name: str) -> VariantSpec:
    """Resolve variant name -> VariantSpec."""
    n = name.upper()
    if n.startswith("V") and n[1:].isdigit():
        return VariantSpec(name=n, family="legacy", input_mode="sequence",
                           kd_mode="per_frame")
    if n.startswith("C1"):
        return VariantSpec(name=n, family="C1", input_mode="sequence",
                           kd_mode="per_frame")
    if n.startswith("C2"):
        return VariantSpec(name=n, family="C2", input_mode="stacked",
                           kd_mode="video_level")
    if n.startswith("C3"):
        return VariantSpec(name=n, family="C3", input_mode="sequence",
                           kd_mode="per_frame")
    raise ValueError(f"unknown variant: {name}")


def build_clip_model(name: str):
    """Returns (clip_model, spec, streaming_model_or_None, extras_dict)."""
    spec = get_spec(name)
    if spec.family == "legacy":
        from student_model import VARIANTS, build_variant
        if name.upper() not in VARIANTS:
            raise ValueError(f"legacy variant {name} not in VARIANTS")
        bundle = build_variant(name.upper())
        # Also fill T/img_size from legacy cfg
        spec.T = bundle["cfg"].T
        spec.img_size = bundle["cfg"].H
        return bundle["clip_model"], spec, bundle.get("streaming_model"), bundle
    if spec.family == "C1":
        from tsm_mbv2_tf import build_tsm_mbv2
        preset = C1_PRESETS[spec.name]
        bundle = build_tsm_mbv2(**preset)
        # Propagate input_hw from preset back into spec so dataloader uses it
        spec_with_hw = VariantSpec(
            name=spec.name, family=spec.family, input_mode=spec.input_mode,
            kd_mode=spec.kd_mode, T=spec.T, img_size=preset["input_hw"],
            in_channels=spec.in_channels,
        )
        return bundle["clip_model"], spec_with_hw, bundle["streaming_model"], bundle
    if spec.family == "C2":
        from path_c2_factory import build_c2
        bundle = build_c2(spec.name)
        return bundle["model"], spec, None, bundle
    if spec.family == "C3":
        from path_c3_factory import build_c3
        bundle = build_c3(spec.name)
        return bundle["clip_model"], spec, bundle["streaming_model"], bundle
    raise ValueError(f"unknown family: {spec.family}")
