"""Day 10 Path C2: Stacked-frame MobileNet (V1 or V2) for video-level classification.

Key idea: collapse the time axis into the channel axis. Input shape:
    [B, H, W, T]   (T=8 grayscale frames stacked as 8 channels)

Pros:
  * Pure 2D ConvNet -- zero TSM ops, trivially deployable as a single TFLite
  * No cache management on U5; one ai_run per clip (40-50ms for 200K param model)
  * Single forward pass, no streaming complexity

Cons:
  * No causal early-exit (must see all 8 frames before predicting)
  * KD uses video-level loss: take TA's last-frame logits (obs=1.0, most
    confident -- TA per-frame[7] argmax matches train label 95.5% per Day 9
    sanity check) as the soft target.

Variants (Plan Day 8-10 Path C matrix):
    C2a: MobileNet V1, width=0.25 (~200K)
    C2b: MobileNet V1, width=0.50 (~800K)
    C2c: MobileNet V2, width=0.35 (~350K)
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model, layers

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mobilenet_blocks_tf import (
    MBV1_INPUT_CHANNELS, MBV1_SETTINGS, MBV2_INPUT_CHANNELS, MBV2_LAST_CHANNELS,
    MBV2_SETTINGS, _make_divisible, conv_stem, dw_separable_block,
    inverted_residual_block,
)


def build_c2_v1(
    n_classes: int = 27,
    input_hw: int = 64,
    T_in: int = 8,
    width_mult: float = 0.25,
    dropout: float = 0.5,
) -> Model:
    """MobileNet V1 with 8-channel input (T stacked frames)."""
    stem_ch = _make_divisible(MBV1_INPUT_CHANNELS * width_mult)
    inp = layers.Input(shape=(input_hw, input_hw, T_in), name="stacked_in")
    x = conv_stem(inp, stem_ch, stride=2, name="stem")
    for i, (out_c, stride) in enumerate(MBV1_SETTINGS):
        out_ch = _make_divisible(out_c * width_mult)
        x = dw_separable_block(x, out_channels=out_ch, stride=stride, name=f"dwblk_{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(
        n_classes, name="head",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
    )(x)
    return Model(inp, logits, name=f"c2_mbv1_w{width_mult}")


def build_c2_v2(
    n_classes: int = 27,
    input_hw: int = 64,
    T_in: int = 8,
    width_mult: float = 0.35,
    dropout: float = 0.5,
) -> Model:
    """MobileNet V2 with 8-channel input (T stacked frames)."""
    stem_ch = _make_divisible(MBV2_INPUT_CHANNELS * width_mult)
    last_ch = _make_divisible(MBV2_LAST_CHANNELS * max(width_mult, 1.0))

    inp = layers.Input(shape=(input_hw, input_hw, T_in), name="stacked_in")
    x = conv_stem(inp, stem_ch, stride=2, name="stem")

    block_idx = 0
    for (t, c_base, n, s) in MBV2_SETTINGS:
        out_ch = _make_divisible(c_base * width_mult)
        for i in range(n):
            stride = s if i == 0 else 1
            x = inverted_residual_block(
                x, out_channels=out_ch, expand_ratio=t, stride=stride,
                name=f"ir_{block_idx}",
            )
            block_idx += 1

    # Final 1x1 conv to last_ch
    x = layers.Conv2D(last_ch, 1, padding="same", use_bias=False, name="head_conv")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.ReLU(6.0, name="head_relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(
        n_classes, name="head_dense",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
    )(x)
    return Model(inp, logits, name=f"c2_mbv2_w{width_mult}")


# ---- Public factory for Path C2 --------------------------------------------

def build_c2(variant: str) -> dict:
    """C2 variants:
        c2a -> MBV1 width=0.25
        c2b -> MBV1 width=0.50
        c2c -> MBV2 width=0.35
    """
    variant = variant.lower()
    if variant == "c2a":
        model = build_c2_v1(width_mult=0.25)
    elif variant == "c2b":
        model = build_c2_v1(width_mult=0.50)
    elif variant == "c2c":
        model = build_c2_v2(width_mult=0.35)
    else:
        raise ValueError(f"unknown C2 variant: {variant}")
    return {"model": model, "input_mode": "stacked",
            "kd_mode": "video_level", "T_in_channels": 8}


# ---- Smoke test -----------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    tf.keras.utils.set_random_seed(0)

    for variant, expect in [
        ("c2a", (100_000, 350_000)),
        ("c2b", (350_000, 1_400_000)),
        ("c2c", (200_000, 600_000)),
    ]:
        d = build_c2(variant)
        m = d["model"]
        n_params = int(sum(int(np.prod(v.shape)) for v in m.trainable_weights))
        # smoke forward
        x = tf.random.normal((2, 64, 64, 8))   # stacked 8-channel input
        y = m(x, training=False)
        assert y.shape.as_list() == [2, 27], y.shape
        print(f"{variant}: {m.name}  params={n_params:,}  output={tuple(y.shape)}")
        lo, hi = expect
        assert lo <= n_params <= hi, f"params {n_params} not in [{lo}, {hi}]"
    print("\nALL C2 SMOKE CHECKS PASSED")
