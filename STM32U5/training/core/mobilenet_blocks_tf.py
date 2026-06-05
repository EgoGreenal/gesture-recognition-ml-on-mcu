"""Day 10 Path C: MobileNet V1/V2 building blocks for TF/Keras.

Used by:
  * tsm_mbv2_tf.py     (C1: TSM-MBV2 truncated)
  * path_c2_factory.py (C2: Stacked-frame MobileNet, video-level KD)
  * path_c3_factory.py (C3: Hybrid stacked-early + TSM-late)

Width_mult convention (matches Keras built-in MobileNet):
    channel_round = max(1, round(base_channels * width_mult / 8) * 8)
divisor=8 for INT8 friendliness on Cortex-M33 + MVE.

Pretrained weights:
  We do NOT load ImageNet pretrained weights because:
    1. Jester input is grayscale 1ch, not RGB 3ch -- first conv shape mismatch
    2. Width_mult variants (0.25 / 0.5 / 0.35) don't have stock pretrained
       checkpoints; would need careful slicing or re-training of stem regardless
  Initialization: HeNormal for Conv, ones/zeros for BN -- standard TF defaults.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


# ---- channel rounding -----------------------------------------------------

def _make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round channels to nearest multiple of `divisor` (default 8).

    Mirrors torchvision.models.mobilenetv2._make_divisible: keeps the rounded
    value within 90% of the original (i.e. no aggressive shrinking).
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


# ---- MobileNet V1 depthwise separable block ------------------------------

def dw_separable_block(
    x: tf.Tensor,
    out_channels: int,
    stride: int = 1,
    name: str = "dw",
    use_bias: bool = False,
    relu_max_value: float = 6.0,
) -> tf.Tensor:
    """DW conv -> BN -> ReLU6 -> PW conv -> BN -> ReLU6 (MobileNet V1).

    out_channels: final channels after PW.
    """
    x = layers.DepthwiseConv2D(
        3, strides=stride, padding="same", use_bias=use_bias, name=f"{name}_dw_conv"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(relu_max_value, name=f"{name}_dw_relu")(x)
    x = layers.Conv2D(
        out_channels, 1, strides=1, padding="same", use_bias=use_bias,
        name=f"{name}_pw_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(relu_max_value, name=f"{name}_pw_relu")(x)
    return x


# ---- MobileNet V2 inverted residual block --------------------------------

def inverted_residual_block(
    x: tf.Tensor,
    out_channels: int,
    expand_ratio: int = 6,
    stride: int = 1,
    name: str = "ir",
    use_bias: bool = False,
    relu_max_value: float = 6.0,
    tsm_layer: layers.Layer | None = None,
) -> tf.Tensor:
    """Inverted Residual block (MobileNet V2).

    Sequence:
      [optional TSM shift -- per MIT TSM, applied on the expansion conv input
       so the time signal propagates through the whole block]
      → 1x1 expansion conv → BN → ReLU6
      → 3x3 depthwise conv (stride) → BN → ReLU6
      → 1x1 projection conv → BN (no activation)
      → optional residual add (only if stride==1 and in_ch == out_ch)

    Args:
        tsm_layer: if not None, applied to `x` before the expansion conv.
                   For Path C1 (per-frame TSM-MBV2 trunc) this is a callable
                   that does the causal shift. For C2 and the early blocks of
                   C3 this is None (pure 2D).
    """
    in_channels = int(x.shape[-1])
    use_res = (stride == 1) and (in_channels == out_channels)
    inp = x

    # Optional TSM injection (matches MIT 'blockres' placement: before expand
    # conv of each residual block).
    if tsm_layer is not None and use_res:
        x = tsm_layer(x)

    # Expansion 1x1
    exp_ch = in_channels * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            exp_ch, 1, strides=1, padding="same", use_bias=use_bias,
            name=f"{name}_expand_conv",
        )(x)
        x = layers.BatchNormalization(name=f"{name}_expand_bn")(x)
        x = layers.ReLU(relu_max_value, name=f"{name}_expand_relu")(x)

    # Depthwise 3x3
    x = layers.DepthwiseConv2D(
        3, strides=stride, padding="same", use_bias=use_bias,
        name=f"{name}_dw_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(relu_max_value, name=f"{name}_dw_relu")(x)

    # Projection 1x1 (linear -- no activation)
    x = layers.Conv2D(
        out_channels, 1, strides=1, padding="same", use_bias=use_bias,
        name=f"{name}_proj_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_proj_bn")(x)

    if use_res:
        x = layers.Add(name=f"{name}_residual")([inp, x])
    return x


# ---- Conv stem ------------------------------------------------------------

def conv_stem(
    x: tf.Tensor,
    out_channels: int,
    stride: int = 2,
    name: str = "stem",
    use_bias: bool = False,
    relu_max_value: float = 6.0,
) -> tf.Tensor:
    """Initial 3x3 standard conv (stride 2) -> BN -> ReLU6."""
    x = layers.Conv2D(
        out_channels, 3, strides=stride, padding="same", use_bias=use_bias,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(relu_max_value, name=f"{name}_relu")(x)
    return x


# ---- MobileNet V2 architecture spec (standard 17 blocks) -----------------
#
# Each row: (t=expand_ratio, c=out_channels, n=#blocks, s=first_block_stride)
# This is the MobileNet V2 paper / torchvision default.

MBV2_SETTINGS = [
    # t,  c,    n,  s
    (1,   16,   1,  1),    # block 0
    (6,   24,   2,  2),    # blocks 1-2
    (6,   32,   3,  2),    # blocks 3-5
    (6,   64,   4,  2),    # blocks 6-9
    (6,   96,   3,  1),    # blocks 10-12
    (6,   160,  3,  2),    # blocks 13-15
    (6,   320,  1,  1),    # block 16  (total 17 blocks)
]
MBV2_INPUT_CHANNELS = 32
MBV2_LAST_CHANNELS = 1280


# ---- MobileNet V1 architecture spec --------------------------------------
#
# MobileNet V1 has 13 depthwise-separable blocks after the stem. Each tuple:
# (out_channels, stride).

MBV1_SETTINGS = [
    (64,   1),
    (128,  2),
    (128,  1),
    (256,  2),
    (256,  1),
    (512,  2),
    (512,  1),  # repeats x5
    (512,  1),
    (512,  1),
    (512,  1),
    (512,  1),
    (1024, 2),
    (1024, 1),
]
MBV1_INPUT_CHANNELS = 32


# ---- Smoke test -----------------------------------------------------------

if __name__ == "__main__":
    tf.keras.utils.set_random_seed(0)

    # Test divisible rounding
    assert _make_divisible(32 * 0.25) == 8
    assert _make_divisible(32 * 0.5) == 16
    # 32*0.35=11.2; rounds to 8 but 8 < 0.9*11.2=10.08 → bumps up to 16
    assert _make_divisible(32 * 0.35) == 16
    print("_make_divisible: OK")

    # Test dw_separable_block
    x = tf.keras.Input((64, 64, 8), name="test")
    y = dw_separable_block(x, out_channels=16, stride=2, name="dw1")
    print(f"dw_separable_block: {tuple(x.shape)} -> {tuple(y.shape)}")
    assert y.shape.as_list() == [None, 32, 32, 16]

    # Test inverted_residual_block (no TSM)
    x = tf.keras.Input((32, 32, 24), name="test2")
    y = inverted_residual_block(x, out_channels=24, expand_ratio=6, stride=1, name="ir1")
    print(f"inverted_residual_block (residual): {tuple(x.shape)} -> {tuple(y.shape)}")
    assert y.shape.as_list() == [None, 32, 32, 24]

    # Test inverted_residual_block (with stride 2, no residual)
    x = tf.keras.Input((32, 32, 16), name="test3")
    y = inverted_residual_block(x, out_channels=24, expand_ratio=6, stride=2, name="ir2")
    print(f"inverted_residual_block (stride=2): {tuple(x.shape)} -> {tuple(y.shape)}")
    assert y.shape.as_list() == [None, 16, 16, 24]

    print("ALL SMOKE CHECKS PASSED")
