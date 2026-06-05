"""Day 10 Path C3: Hybrid -- stacked-frame 2D early + TSM late.

Architecture:
    Input [B, T=8, H=64, W=64, 1]
        |  TimeDistributed early stage (N_spatial 2D blocks, no temporal mixing)
        |  -- per-frame feature maps, spatial downsample to ~16x16 or 8x8
        v
    Mid feature [B, T, H', W', C']
        |  Now temporal modeling via M TSM-injected IR blocks (TSM at block input)
        v
    Late feature [B, T, H'', W'', C'']
        |  TimeDistributed head -- per-frame logits (KD: per-frame on TSM tail)
        v
    Output [B, T, 27]

Rationale (Plan Day 8):
  * 2D early stage = cheaper than TSM early; depthwise separable enough to learn
    per-frame appearance features
  * TSM late stage = applies temporal mixing only at smaller spatial sizes,
    fewer ops + fewer cache buffers on U5
  * Best of both: deployment cost halfway between C1 (full TSM) and C2 (no TSM)

Variants:
    C3a: 3 spatial + 2 TSM @ 16x16  (~250K)
    C3b: 4 spatial + 3 TSM @ 8x8    (~350K)
    C3c: 5 spatial + 2 TSM @ 4x4    (~280K)
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model, layers

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mobilenet_blocks_tf import (
    _make_divisible, conv_stem, dw_separable_block, inverted_residual_block,
)
from tsm_streaming_layer import TSMStreamingShift, causal_tsm_clip


def _build_2d_spatial_submodel(
    in_shape: tuple,
    out_channels: int,
    stride: int,
    name: str,
) -> Model:
    """One spatial block (DW separable 2D) as a Functional sub-model.

    in_shape: (H, W, C_in)
    """
    inp = layers.Input(shape=in_shape, name=f"{name}_in")
    x = dw_separable_block(inp, out_channels=out_channels, stride=stride, name=name)
    return Model(inp, x, name=f"{name}_submodel")


def _build_tsm_ir_submodel(
    in_shape: tuple,
    out_channels: int,
    expand_ratio: int,
    stride: int,
    name: str,
) -> Model:
    """Inverted Residual block as Functional sub-model. TSM applied OUTSIDE."""
    inp = layers.Input(shape=in_shape, name=f"{name}_in")
    out = inverted_residual_block(
        inp, out_channels=out_channels, expand_ratio=expand_ratio,
        stride=stride, name=name, tsm_layer=None,
    )
    return Model(inp, out, name=f"{name}_submodel")


def build_c3(
    variant: str,
    T: int = 8,
    input_hw: int = 64,
    in_channels: int = 1,
    n_classes: int = 27,
    shift_div: int = 8,
    dropout: float = 0.5,
) -> dict:
    """C3 variants share the same architectural template; differ in
    (n_spatial, n_tsm, spatial_at_tsm_split, channel ladder)."""
    variant = variant.lower()
    # TSM injection requires stride=1 AND in_ch == out_ch (use_res_connect).
    # So spatial downsampling must finish BEFORE the TSM stack.
    if variant == "c3a":
        # 3 spatial blocks bring 64x64 -> 16x16, ending at 96 channels.
        # 2 TSM blocks @ 16x16 (stride=1, channels stay 96).
        spatial_spec = [(24, 2), (48, 2), (96, 1)]
        tsm_spec = [(96, 6, 1), (96, 6, 1)]
    elif variant == "c3b":
        # 4 spatial blocks bring 64x64 -> 8x8 (3 stride-2 + 1 stride-1).
        # 3 TSM @ 8x8 (channels stay 96).
        spatial_spec = [(24, 2), (48, 2), (72, 2), (96, 1)]
        tsm_spec = [(96, 6, 1), (96, 6, 1), (96, 6, 1)]
    elif variant == "c3c":
        # 5 spatial blocks bring 64x64 -> 4x4 (4 stride-2 + 1 stride-1).
        # 2 TSM @ 4x4 (channels stay 128).
        spatial_spec = [(24, 2), (48, 2), (72, 2), (96, 2), (128, 1)]
        tsm_spec = [(128, 6, 1), (128, 6, 1)]
    else:
        raise ValueError(f"unknown C3 variant: {variant}")

    # ---- Build sub-models (shared between clip + streaming) ----
    spatial_blocks: list[Model] = []
    h_cur, w_cur, c_cur = input_hw, input_hw, in_channels
    for i, (out_ch, stride) in enumerate(spatial_spec):
        block = _build_2d_spatial_submodel(
            in_shape=(h_cur, w_cur, c_cur),
            out_channels=out_ch, stride=stride, name=f"sp{i}",
        )
        spatial_blocks.append(block)
        h_cur = (h_cur + stride - 1) // stride
        w_cur = (w_cur + stride - 1) // stride
        c_cur = out_ch

    # ---- Build TSM-IR blocks ----
    tsm_blocks: list[Model] = []
    tsm_streaming_layers: list[TSMStreamingShift] = []
    tsm_block_input_dims: list[tuple] = []   # (H_in, W_in, C_in, n_shift)
    for i, (out_ch, expand, stride) in enumerate(tsm_spec):
        # TSM applied if cur_ch divisible by shift_div AND use_res holds
        use_res = (stride == 1) and (c_cur == out_ch)
        has_tsm = use_res and (c_cur % shift_div == 0)
        if has_tsm:
            tsm_streaming_layers.append(TSMStreamingShift(
                n_div=shift_div, name=f"tsm_stream_t{i}",
            ))
            tsm_block_input_dims.append((h_cur, w_cur, c_cur, c_cur // shift_div))
        else:
            tsm_streaming_layers.append(None)
            tsm_block_input_dims.append(None)
        block = _build_tsm_ir_submodel(
            in_shape=(h_cur, w_cur, c_cur),
            out_channels=out_ch, expand_ratio=expand, stride=stride,
            name=f"t{i}",
        )
        tsm_blocks.append(block)
        h_cur = (h_cur + stride - 1) // stride
        w_cur = (w_cur + stride - 1) // stride
        c_cur = out_ch

    # ---- Head (per-frame logits) ----
    head_in = layers.Input(shape=(h_cur, w_cur, c_cur), name="head_in")
    h = layers.GlobalAveragePooling2D(name="head_gap")(head_in)
    h = layers.Dropout(dropout, name="head_dropout")(h)
    h = layers.Dense(
        n_classes, name="head_dense",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
    )(h)
    head_submodel = Model(head_in, h, name="head_submodel")

    # ---- Clip-form graph (training) ----
    # Input (B, T, H, W, C) -> output (B, T, n_classes)
    # Early spatial blocks: TimeDistributed (no temporal interaction)
    clip_in = layers.Input(shape=(T, input_hw, input_hw, in_channels), name="clip")
    x = clip_in
    for i, block in enumerate(spatial_blocks):
        x = layers.TimeDistributed(block, name=f"td_sp{i}")(x)
    # TSM-IR blocks: insert causal_tsm_clip before block if has_tsm
    for i, block in enumerate(tsm_blocks):
        if tsm_streaming_layers[i] is not None:
            x = layers.Lambda(
                lambda t, n=shift_div: causal_tsm_clip(t, n_div=n),
                name=f"tsm_clip_t{i}",
            )(x)
        x = layers.TimeDistributed(block, name=f"td_t{i}")(x)
    logits = layers.TimeDistributed(head_submodel, name="td_head")(x)
    clip_model = Model(clip_in, logits, name=f"c3_{variant}_clip")

    # ---- Streaming-form graph (deployment) ----
    frame_in = layers.Input(shape=(input_hw, input_hw, in_channels),
                            batch_size=1, name="frame_in")
    x = frame_in
    # Early 2D pass-through
    for block in spatial_blocks:
        x = block(x)
    # TSM blocks with cache I/O
    cache_in_keras = []
    cache_out_tensors = []
    for i, (block, dim) in enumerate(zip(tsm_blocks, tsm_block_input_dims)):
        if dim is not None:
            H_in, W_in, C_in, n_shift = dim
            c_in = layers.Input(shape=(H_in, W_in, n_shift), batch_size=1,
                                name=f"cache_t{i}_in")
            cache_in_keras.append(c_in)
            stream_layer = tsm_streaming_layers[i]
            x, c_out = stream_layer([x, c_in])
            cache_out_tensors.append(c_out)
        x = block(x)
    logits = head_submodel(x)
    streaming_model = Model(
        inputs=[frame_in] + cache_in_keras,
        outputs=[logits] + cache_out_tensors,
        name=f"c3_{variant}_streaming",
    )

    cfg = dict(
        variant=variant, T=T, input_hw=input_hw, in_channels=in_channels,
        n_classes=n_classes, shift_div=shift_div,
        spatial_spec=spatial_spec, tsm_spec=tsm_spec,
        n_tsm_sites=sum(1 for d in tsm_block_input_dims if d is not None),
        final_spatial=(h_cur, w_cur), final_channels=c_cur,
    )
    return {
        "cfg": cfg,
        "clip_model": clip_model,
        "streaming_model": streaming_model,
        "spatial_blocks": spatial_blocks,
        "tsm_blocks": tsm_blocks,
        "head": head_submodel,
        "input_mode": "sequence",
        "kd_mode": "per_frame_tsm_tail",
    }


# ---- Smoke test -----------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    tf.keras.utils.set_random_seed(0)

    for variant, expect in [
        ("c3a", (50_000, 600_000)),
        ("c3b", (50_000, 800_000)),
        ("c3c", (50_000, 800_000)),
    ]:
        d = build_c3(variant)
        m = d["clip_model"]
        n_params = int(sum(int(np.prod(v.shape)) for v in m.trainable_weights))
        x = tf.random.normal((1, 8, 64, 64, 1))
        y = m(x, training=False)
        assert y.shape.as_list() == [1, 8, 27]
        print(f"{variant}: params={n_params:,}  tsm_sites={d['cfg']['n_tsm_sites']}  "
              f"final={d['cfg']['final_spatial']}x{d['cfg']['final_channels']}ch  "
              f"output={tuple(y.shape)}")
        # Streaming forward
        f = tf.zeros((1, 64, 64, 1))
        caches = []
        for inp in d["streaming_model"].inputs:
            if inp.name.startswith("cache_"):
                caches.append(tf.zeros(inp.shape.as_list()))
        out = d["streaming_model"]([f] + caches, training=False)
        assert out[0].shape.as_list() == [1, 27]
        lo, hi = expect
        assert lo <= n_params <= hi, f"params {n_params} not in [{lo}, {hi}]"
    print("\nALL C3 SMOKE CHECKS PASSED")
