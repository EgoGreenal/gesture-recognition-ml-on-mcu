"""Day 9: Student model factory — V0..V6 with dual export (clip-form + streaming-form).

Per Project_Plan_ch.md Section 5 + Day 8 handoff:

  * 7 variants (V0..V6), each parameterized by (H, W, T, channels[], strides[], conv_type, n_div)
  * Block primitive: ``Conv(stride) → BN → ReLU6`` (DW + PW pair if conv_type='dw').
  * TSM cut placed AFTER ReLU of every block EXCEPT the last block (matches the
    Day 8 probe layout — last stage feeds GAP + head, no cache I/O after it).
  * Two graphs share weights:
      - **clip-form** (training): input ``(B, T, H, W, 1)``, output ``(B, T, K)``
        per-frame logits via ``causal_tsm_clip`` (rank-5 T-axis shift). Fully
        differentiable, batched, fast on GPU.
      - **streaming-form** (deployment): input single frame + N cache tensors,
        output logits + N cache tensors. Uses ``TSMStreamingShift`` layer
        which the Day 8 probe verified is quantize-clean + stedgeai-accepted.
  * The two graphs reference the **same block sub-models** and the **same head
    Dense layer** — so saving weights from clip-form and rebuilding streaming-
    form gives bit-identical math.

Plan Section 5 variant table (channels chosen so ``ch % n_div == 0`` at every
TSM boundary; n_div lowered to 4 for V0/V4 since their "base ch=12" is not a
multiple of 8):

    V0 Ultra-Tiny:  64x64x1, T=8, ch=[12,24,48],          stride=[2,2,2], dw,  n_div=4
    V1 Tiny:        48x48x1, T=6, ch=[8,16,32],           stride=[2,2,2], std, n_div=8
    V2 Small:       64x64x1, T=8, ch=[16,32,64],          stride=[2,2,2], std, n_div=8
    V3 Small-DW:    64x64x1, T=8, ch=[16,32,32,64],       stride=[2,2,1,2], dw, n_div=8
    V4 Deep:        64x64x1, T=8, ch=[12,24,24,48,48],    stride=[2,1,2,1,2], std, n_div=4
    V5 Wide:        64x64x1, T=8, ch=[32,64,128],         stride=[2,2,2], std, n_div=8
    V6 HiRes:       96x96x1, T=8, ch=[16,32,64],          stride=[2,2,2], dw,  n_div=8
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model, layers

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tsm_streaming_layer import TSMStreamingShift, causal_tsm_clip


# ---- Variant config --------------------------------------------------------

@dataclass(frozen=True)
class VariantConfig:
    name: str
    H: int
    W: int
    T: int                       # frames seen by student
    channels: tuple              # output channel count of each block
    strides: tuple               # stride of each block (3 or 1, 2)
    conv_type: str = "std"       # 'std' or 'dw'
    n_div: int = 8               # TSM channel divisor (n_shift = C // n_div)
    n_classes: int = 27
    in_channels: int = 1         # grayscale
    obs_ratios: tuple = (0.25, 0.50, 0.75, 1.0)


VARIANTS: dict[str, VariantConfig] = {
    "V0": VariantConfig("V0", 64, 64, 8, (12, 24, 48),            (2, 2, 2),       "dw",  4),
    "V1": VariantConfig("V1", 48, 48, 6, (8, 16, 32),             (2, 2, 2),       "std", 8),
    "V2": VariantConfig("V2", 64, 64, 8, (16, 32, 64),            (2, 2, 2),       "std", 8),
    "V3": VariantConfig("V3", 64, 64, 8, (16, 32, 32, 64),        (2, 2, 1, 2),    "dw",  8),
    "V4": VariantConfig("V4", 64, 64, 8, (12, 24, 24, 48, 48),    (2, 1, 2, 1, 2), "std", 4),
    "V5": VariantConfig("V5", 64, 64, 8, (32, 64, 128),           (2, 2, 2),       "std", 8),
    "V6": VariantConfig("V6", 96, 96, 8, (16, 32, 64),            (2, 2, 2),       "dw",  8),
    # ---------------------------------------------------------------------
    # Day 10 v3: depth scan after V4 (5-block, 38.45%) ≫ V5 (3-block, 25%)
    # confirmed that TSM needs temporal depth to aggregate the 8-frame window.
    # V7/V8/V9 sweep depth at 6/7/8 blocks, std conv, V4-style channel ladder.
    # ---------------------------------------------------------------------
    "V7": VariantConfig("V7", 64, 64, 8, (12, 24, 24, 48, 48, 96),
                                                                  (2, 1, 2, 1, 2, 1),    "std", 4),
    "V8": VariantConfig("V8", 64, 64, 8, (12, 24, 24, 48, 48, 96, 96),
                                                                  (2, 1, 2, 1, 2, 1, 1), "std", 4),
    "V9": VariantConfig("V9", 64, 64, 8, (12, 24, 24, 48, 48, 96, 96, 96),
                                                                  (2, 1, 2, 1, 2, 1, 1, 1), "std", 4),
}


def student_frame_indices(T: int, obs_ratios=(0.25, 0.5, 0.75, 1.0)) -> list[int]:
    """0-indexed frame positions to supervise at each observation ratio.

    Pattern matches Plan Day 8: for T=8, obs ratios {0.25, 0.5, 0.75, 1.0}
    → frames [1, 3, 5, 7].
    For T=6: [1, 2, 4, 5] (close to plan's "V1 supervised frames [2,3,5,6]"
    in 1-indexed form).
    """
    return [max(int(round(T * r)) - 1, 0) for r in obs_ratios]


# ---- Block sub-model (reusable across clip + streaming) --------------------

def _block_submodel(cfg: VariantConfig, idx: int, in_shape: tuple, out_ch: int,
                    stride: int) -> Model:
    """Conv-BN-ReLU6 (or DW-BN-ReLU6 + PW-BN-ReLU6 for 'dw'). No TSM inside.

    Layer naming is deterministic — clip and streaming graphs both call this
    same sub-model so weights are shared by reference.
    """
    inp = layers.Input(shape=in_shape, name=f"b{idx}_in")
    x = inp
    if cfg.conv_type == "dw":
        # MobileNet-style depthwise separable block
        x = layers.DepthwiseConv2D(
            3, strides=stride, padding="same", use_bias=False, name=f"b{idx}_dw"
        )(x)
        x = layers.BatchNormalization(name=f"b{idx}_dw_bn")(x)
        x = layers.ReLU(6.0, name=f"b{idx}_dw_relu")(x)
        x = layers.Conv2D(
            out_ch, 1, strides=1, padding="same", use_bias=False, name=f"b{idx}_pw"
        )(x)
        x = layers.BatchNormalization(name=f"b{idx}_pw_bn")(x)
        x = layers.ReLU(6.0, name=f"b{idx}_pw_relu")(x)
    elif cfg.conv_type == "std":
        x = layers.Conv2D(
            out_ch, 3, strides=stride, padding="same", use_bias=False,
            name=f"b{idx}_conv",
        )(x)
        x = layers.BatchNormalization(name=f"b{idx}_bn")(x)
        x = layers.ReLU(6.0, name=f"b{idx}_relu")(x)
    else:
        raise ValueError(f"conv_type must be 'std' or 'dw', got {cfg.conv_type!r}")
    return Model(inp, x, name=f"block_{idx}")


def _stem(cfg: VariantConfig) -> tuple[list[Model], layers.Layer]:
    """Build the shared layer stack: list of block sub-models + the head Dense.

    Returns:
        blocks: list[Model] of length len(cfg.channels)
        head:   shared Dense layer (logits, no activation)
    """
    blocks: list[Model] = []
    H_cur, W_cur, C_cur = cfg.H, cfg.W, cfg.in_channels
    for i, (out_ch, stride) in enumerate(zip(cfg.channels, cfg.strides)):
        in_shape = (H_cur, W_cur, C_cur)
        block = _block_submodel(cfg, i, in_shape, out_ch, stride)
        blocks.append(block)
        # 'same' padding: H_out = ceil(H_in / stride)
        H_cur = (H_cur + stride - 1) // stride
        W_cur = (W_cur + stride - 1) // stride
        C_cur = out_ch
    head = layers.Dense(cfg.n_classes, name="head")
    return blocks, head


# ---- Clip-form (training) --------------------------------------------------

def build_clip_model(cfg: VariantConfig, blocks: list[Model] | None = None,
                     head: layers.Layer | None = None) -> tuple[Model, list[Model], layers.Layer]:
    """Build clip-form model. Returns (model, blocks, head).

    Forward:
        x in (B, T, H, W, 1)
        for i in range(n_blocks):
            x = TimeDistributed(blocks[i])(x)                  # (B, T, H', W', C')
            if i < n_blocks - 1:
                x = causal_tsm_clip(x, n_div=cfg.n_div)
        x = TimeDistributed(GlobalAveragePooling2D)(x)         # (B, T, C)
        logits = TimeDistributed(head)(x)                       # (B, T, K)
    """
    if blocks is None or head is None:
        blocks, head = _stem(cfg)

    clip_in = layers.Input(shape=(cfg.T, cfg.H, cfg.W, cfg.in_channels), name="clip")
    x = clip_in
    n_blocks = len(blocks)
    for i, block in enumerate(blocks):
        x = layers.TimeDistributed(block, name=f"td_block_{i}")(x)
        if i < n_blocks - 1:
            n_div = cfg.n_div  # closure-captured
            x = layers.Lambda(
                lambda t, n=n_div: causal_tsm_clip(t, n_div=n),
                name=f"tsm_clip_{i}",
            )(x)
    gap = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name="td_gap")(x)
    logits = layers.TimeDistributed(head, name="td_head")(gap)
    model = Model(clip_in, logits, name=f"student_{cfg.name}_clip")
    return model, blocks, head


# ---- Streaming-form (deployment) -------------------------------------------

def build_streaming_model(cfg: VariantConfig, blocks: list[Model],
                          head: layers.Layer) -> Model:
    """Build streaming-form model with multi-I/O cache interface.

    Inputs:
        frame_in:    (1, H, W, 1)
        cache_i_in:  (1, H_i, W_i, C_i // n_div)  for each TSM cut (i = 0..n_blocks-2)

    Outputs:
        logits:      (1, K)
        cache_i_out: (1, H_i, W_i, C_i // n_div)  for each TSM cut
    """
    frame_in = layers.Input(shape=(cfg.H, cfg.W, cfg.in_channels),
                            batch_size=1, name="frame_in")
    # Compute spatial sizes after each block (for cache shapes)
    H_cur, W_cur = cfg.H, cfg.W
    n_blocks = len(blocks)
    cache_ins: list[tf.Tensor] = []
    cache_in_keras: list[layers.Input] = []
    block_out_spatial = []  # (H_after_block, W_after_block) for cuts
    for i, stride in enumerate(cfg.strides):
        H_cur = (H_cur + stride - 1) // stride
        W_cur = (W_cur + stride - 1) // stride
        block_out_spatial.append((H_cur, W_cur))
    # Build cache_in inputs for cuts i in 0..n_blocks-2 (no TSM after last)
    for i in range(n_blocks - 1):
        H_i, W_i = block_out_spatial[i]
        n_shift = cfg.channels[i] // cfg.n_div
        c_in = layers.Input(shape=(H_i, W_i, n_shift), batch_size=1,
                            name=f"cache_{i}_in")
        cache_in_keras.append(c_in)
        cache_ins.append(c_in)

    x = frame_in
    cache_outs: list[tf.Tensor] = []
    for i, block in enumerate(blocks):
        x = block(x)
        if i < n_blocks - 1:
            x, cache_out = TSMStreamingShift(n_div=cfg.n_div, name=f"tsm_stream_{i}")(
                [x, cache_in_keras[i]]
            )
            cache_outs.append(cache_out)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    logits = head(x)

    model = Model(
        inputs=[frame_in] + cache_in_keras,
        outputs=[logits] + cache_outs,
        name=f"student_{cfg.name}_streaming",
    )
    return model


# ---- Public factory --------------------------------------------------------

def build_variant(name_or_cfg) -> dict:
    """Build both clip + streaming forms of a variant.

    Returns dict with keys:
        cfg:              VariantConfig
        clip_model:       Keras Model — for training (input (B,T,H,W,1))
        streaming_model:  Keras Model — for deployment (single-frame + N cache I/O)
        blocks:           list of block sub-models (shared between clip + streaming)
        head:             shared Dense layer
    """
    if isinstance(name_or_cfg, str):
        cfg = VARIANTS[name_or_cfg]
    else:
        cfg = name_or_cfg
    clip_model, blocks, head = build_clip_model(cfg)
    streaming_model = build_streaming_model(cfg, blocks, head)
    return {
        "cfg": cfg,
        "clip_model": clip_model,
        "streaming_model": streaming_model,
        "blocks": blocks,
        "head": head,
    }


# ---- Param count helper ----------------------------------------------------

def variant_param_count(name_or_cfg) -> dict:
    """Return param/MAC estimates without building twice.

    MACs are estimated per single frame (not per clip) — multiply by T to get
    per-clip MACs.
    """
    d = build_variant(name_or_cfg)
    cfg = d["cfg"]
    clip_model = d["clip_model"]
    params = int(sum(tf.size(v).numpy() for v in clip_model.trainable_variables))
    return {
        "variant": cfg.name,
        "params": params,
        "H": cfg.H, "W": cfg.W, "T": cfg.T,
        "channels": list(cfg.channels),
        "conv_type": cfg.conv_type,
        "n_div": cfg.n_div,
    }


# ---- CLI dump --------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all",
                    help="V0..V6 or 'all' (default)")
    ap.add_argument("--summary", action="store_true",
                    help="print Keras model.summary() for selected variant(s)")
    args = ap.parse_args()

    targets = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]
    rows = []
    for name in targets:
        cfg = VARIANTS[name]
        d = build_variant(name)
        clip_m, stream_m = d["clip_model"], d["streaming_model"]
        n_params = int(sum(int(np.prod(w.shape)) for w in clip_m.trainable_weights))
        # Smoke: forward a random clip
        x = tf.random.normal((2, cfg.T, cfg.H, cfg.W, cfg.in_channels))
        y = clip_m(x, training=False)
        assert y.shape.as_list() == [2, cfg.T, cfg.n_classes], \
            f"{name}: unexpected clip-form output shape {y.shape}"
        # Smoke: forward a single frame through streaming model
        frame = tf.zeros((1, cfg.H, cfg.W, cfg.in_channels))
        caches = []
        H_cur, W_cur = cfg.H, cfg.W
        for i, stride in enumerate(cfg.strides):
            H_cur = (H_cur + stride - 1) // stride
            W_cur = (W_cur + stride - 1) // stride
            if i < len(cfg.strides) - 1:
                n_shift = cfg.channels[i] // cfg.n_div
                caches.append(tf.zeros((1, H_cur, W_cur, n_shift)))
        streaming_outs = stream_m([frame] + caches, training=False)
        assert streaming_outs[0].shape.as_list() == [1, cfg.n_classes]
        rows.append({
            "variant": name,
            "params": n_params,
            "HxW": f"{cfg.H}x{cfg.W}", "T": cfg.T,
            "n_blocks": len(cfg.channels),
            "channels": list(cfg.channels),
            "conv_type": cfg.conv_type, "n_div": cfg.n_div,
            "n_cache": len(cfg.channels) - 1,
        })
        if args.summary:
            print(f"\n>>> Variant {name} clip-form\n")
            clip_m.summary(line_length=110, print_fn=print)
            print(f"\n>>> Variant {name} streaming-form\n")
            stream_m.summary(line_length=110, print_fn=print)
    print("\n=== Student variant summary ===")
    print(f"{'V':<4}{'HxW':<10}{'T':<4}{'#blk':<6}{'conv':<6}{'n_div':<6}"
          f"{'channels':<24}{'#cache':<7}{'params':>10}")
    for r in rows:
        print(f"{r['variant']:<4}{r['HxW']:<10}{r['T']:<4}"
              f"{r['n_blocks']:<6}{r['conv_type']:<6}{r['n_div']:<6}"
              f"{str(r['channels']):<24}{r['n_cache']:<7}{r['params']:>10,}")
