"""Day 10 Path C1: TSM-MobileNetV2 truncated, TF/Keras port from PyTorch TA.

Reference PyTorch impl: scripts/ta_tsm_mbv2.py (used to train the TA at 93.12% val).

Key alignment requirements (Pit Hole #1 in Plan Day 10):
  * uni-directional causal shift (NOT MIT's bi-directional)
  * shift_div=8 (same fold size)
  * TSM injected at the START of every residual block (use_res_connect=True),
    i.e. before the expansion 1x1 conv. PyTorch does this on `conv[0]`.
  * Per-frame logits head (no temporal averaging) -- output `[B, T, 27]`

Differences from TA:
  * Width_mult parameter: 0.25 / 0.5 / 0.75 / 1.0 (TA uses 1.0)
  * Block count truncation: n_blocks=10 vs 17 (TA uses 17)
  * Grayscale 1-channel input (TA uses RGB 3-channel) -- no ImageNet pretrain

Two graphs (clip-form for training, streaming-form for U5 deployment) share
the same Keras layer instances so weights transfer for free at inference time.

  * clip-form: [B, T, H, W, 1] -> [B, T, 27] via causal_tsm_clip in time axis
  * streaming-form: 1 frame in + N cache_in -> logits + N cache_out
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model, layers

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mobilenet_blocks_tf import (
    MBV2_INPUT_CHANNELS, MBV2_LAST_CHANNELS, MBV2_SETTINGS, _make_divisible,
    conv_stem, inverted_residual_block,
)
from tsm_streaming_layer import TSMStreamingShift, causal_tsm_clip


# ---- TSM clip-form wrapper that registers an injection-site name ----------
#
# Each TSM injection produces (a) a Lambda that does causal_tsm_clip on the
# (B, T, ...) tensor for training (b) a TSMStreamingShift instance (cache I/O)
# that the streaming graph uses with the SAME weights (TSM has no weights, so
# this is automatic).

class _TSMSite:
    """Per-site bookkeeping: one TSM injection."""
    def __init__(self, name: str, channels: int, n_div: int):
        self.name = name
        self.channels = channels
        self.n_div = n_div
        # streaming-form layer (cache I/O), created on demand
        self._streaming_layer: TSMStreamingShift | None = None

    @property
    def n_shift(self) -> int:
        return self.channels // self.n_div

    def streaming_layer(self) -> TSMStreamingShift:
        if self._streaming_layer is None:
            self._streaming_layer = TSMStreamingShift(
                n_div=self.n_div, name=f"tsm_stream_{self.name}",
            )
        return self._streaming_layer


# ---- Build the trunk (shared between clip and streaming graphs) -----------

def _truncated_mbv2_settings(n_blocks: int):
    """Return a prefix of MBV2_SETTINGS that covers exactly `n_blocks` blocks.

    Standard MBV2 has 17 blocks (sum of `n` in MBV2_SETTINGS = 1+2+3+4+3+3+1).
    For n_blocks < 17, we truncate by reducing the LAST stage's `n` first; if
    that drops to 0, we drop the whole stage and start truncating the
    previous one. This preserves the spatial resolution ladder and avoids
    leaving a no-block stage in the middle.
    """
    total_default = sum(r[2] for r in MBV2_SETTINGS)
    if n_blocks >= total_default:
        return list(MBV2_SETTINGS)
    out = []
    remaining = n_blocks
    for (t, c, n, s) in MBV2_SETTINGS:
        if remaining >= n:
            out.append((t, c, n, s))
            remaining -= n
        elif remaining > 0:
            out.append((t, c, remaining, s))
            remaining = 0
            break
        else:
            break
    return out


def build_tsm_mbv2(
    n_classes: int = 27,
    T: int = 8,
    input_hw: int = 64,
    in_channels: int = 1,
    width_mult: float = 1.0,
    n_blocks: int = 17,
    shift_div: int = 8,
    dropout: float = 0.5,
) -> dict:
    """Build clip-form + streaming-form TSM-MBV2 sharing weights.

    Returns:
        {'cfg': dict, 'clip_model': Model, 'streaming_model': Model,
         'tsm_sites': list[_TSMSite]}
    """
    settings = _truncated_mbv2_settings(n_blocks)
    stem_ch = _make_divisible(MBV2_INPUT_CHANNELS * width_mult)
    last_ch = _make_divisible(MBV2_LAST_CHANNELS * max(width_mult, 1.0))

    # =================== Build layer instances ONCE ===================
    # We don't use Sequential -- we manually construct each layer and call it
    # in two different graphs (clip-form unrolls along T; streaming-form
    # operates on 1 frame + caches).
    #
    # In TF/Keras we just call each `inverted_residual_block` function -- but
    # since we need to share layers across two graphs, we build a sub-model
    # for each block (Functional subgraph) that the outer graph composes.

    # First, materialize the spec into a flat list of (t, c, stride) per block
    flat_spec = []
    for (t, c_base, n, s) in settings:
        for i in range(n):
            stride = s if i == 0 else 1
            flat_spec.append((t, _make_divisible(c_base * width_mult), stride))

    # Build stem sub-model
    stem_in = layers.Input(shape=(input_hw, input_hw, in_channels), name="stem_in")
    stem_out = conv_stem(stem_in, stem_ch, stride=2, name="stem")
    stem_submodel = Model(stem_in, stem_out, name="block_stem")

    # Build each IR block as a sub-model, recording TSM site info.
    blocks: list[Model] = []
    tsm_sites: list[_TSMSite] = []
    cur_ch = stem_ch
    cur_h = input_hw // 2
    for i, (t, out_ch, stride) in enumerate(flat_spec):
        block_in_shape = (cur_h, cur_h, cur_ch)
        bi = layers.Input(shape=block_in_shape, name=f"block_{i}_in")

        # Determine if this block has residual connection (TSM injection site)
        use_res = (stride == 1) and (cur_ch == out_ch)
        if use_res and (cur_ch % shift_div == 0):
            site = _TSMSite(name=f"b{i}", channels=cur_ch, n_div=shift_div)
            tsm_sites.append(site)
            tsm_site_for_block = site
        else:
            tsm_site_for_block = None

        # For the SUB-MODEL we don't actually run TSM here (we handle TSM
        # OUTSIDE the block in the outer graph). The sub-model is the pure
        # 2D inverted residual; TSM is applied to its input by the outer
        # graph. This decoupling is critical because clip-form needs TSM on
        # the (B,T,...) tensor (operates along T) while streaming-form needs
        # TSMStreamingShift (cache I/O).
        bo = inverted_residual_block(
            bi, out_channels=out_ch, expand_ratio=t, stride=stride,
            name=f"block_{i}", tsm_layer=None,
        )
        blocks.append(Model(bi, bo, name=f"block_{i}_submodel"))

        # Update spatial dim for next block
        cur_ch = out_ch
        cur_h = (cur_h + stride - 1) // stride

    # Head: 1x1 conv to last_ch, BN, ReLU6, GAP, dropout, Dense
    head_in_shape = (cur_h, cur_h, cur_ch)
    head_in = layers.Input(shape=head_in_shape, name="head_in")
    h = layers.Conv2D(last_ch, 1, padding="same", use_bias=False, name="head_conv")(head_in)
    h = layers.BatchNormalization(name="head_bn")(h)
    h = layers.ReLU(6.0, name="head_relu")(h)
    h = layers.GlobalAveragePooling2D(name="head_gap")(h)
    h = layers.Dropout(dropout, name="head_dropout")(h)
    h_logits = layers.Dense(n_classes, name="head_dense",
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))(h)
    head_submodel = Model(head_in, h_logits, name="head_submodel")

    # =================== Clip-form graph (training) ===================
    # Input (B, T, H, W, C) -> output (B, T, n_classes)
    clip_in = layers.Input(shape=(T, input_hw, input_hw, in_channels), name="clip")
    # Apply stem time-distributed (it's a Functional sub-model, no time-aware ops)
    x = layers.TimeDistributed(stem_submodel, name="td_stem")(clip_in)

    # Walk through blocks: for each, optionally apply TSM clip-form FIRST,
    # then the IR block.
    site_iter = iter(tsm_sites)
    next_site = next(site_iter, None)
    for i, block in enumerate(blocks):
        # Re-check whether this block had a TSM site (use_res + divisibility)
        # Order is the same as how we built tsm_sites, so consume sequentially.
        block_input_ch = int(block.input_shape[-1])
        use_res = (
            block.output_shape == block.input_shape
        )
        if use_res and (block_input_ch % shift_div == 0) and next_site is not None:
            site = next_site
            next_site = next(site_iter, None)
            # Apply causal_tsm_clip along T axis. Lambda captures site.n_div.
            n_div_local = site.n_div
            x = layers.Lambda(
                lambda t, n=n_div_local: causal_tsm_clip(t, n_div=n),
                name=f"tsm_clip_{site.name}",
            )(x)
        x = layers.TimeDistributed(block, name=f"td_block_{i}")(x)

    # Head time-distributed
    x_gap_logits = layers.TimeDistributed(head_submodel, name="td_head")(x)
    clip_model = Model(clip_in, x_gap_logits, name="c1_tsmmbv2_clip")

    # =================== Streaming-form graph (deployment) ============
    # Input: frame_in [1, H, W, C], cache_*_in [1, H_i, W_i, n_shift_i]
    # Output: logits [1, n_classes], cache_*_out (one per TSM site)
    frame_in = layers.Input(shape=(input_hw, input_hw, in_channels),
                            batch_size=1, name="frame_in")
    # Compute per-block input spatial size (same logic as during build)
    # Track which blocks have TSM sites and their spatial dims
    cache_in_keras: list[layers.Input] = []
    # Walk blocks, find block input shapes for cache_in
    h_cur = input_hw // 2
    c_cur = stem_ch
    block_dims = []
    for (t, out_ch, stride) in flat_spec:
        block_dims.append((c_cur, h_cur))
        c_cur = out_ch
        h_cur = (h_cur + stride - 1) // stride

    # Generate cache_in inputs (one per TSM site)
    site_block_idx = []
    site_idx = 0
    for i, block in enumerate(blocks):
        block_input_ch = int(block.input_shape[-1])
        use_res = (block.output_shape == block.input_shape)
        if use_res and (block_input_ch % shift_div == 0):
            in_ch, in_h = block_dims[i]
            site = tsm_sites[site_idx]
            n_shift = in_ch // site.n_div
            c_in = layers.Input(shape=(in_h, in_h, n_shift), batch_size=1,
                                name=f"cache_{site.name}_in")
            cache_in_keras.append(c_in)
            site_block_idx.append(i)
            site_idx += 1

    # Forward
    x = stem_submodel(frame_in)
    cache_outs: list[tf.Tensor] = []
    site_pos = 0
    for i, block in enumerate(blocks):
        if site_pos < len(site_block_idx) and i == site_block_idx[site_pos]:
            site = tsm_sites[site_pos]
            stream_layer = site.streaming_layer()
            x, c_out = stream_layer([x, cache_in_keras[site_pos]])
            cache_outs.append(c_out)
            site_pos += 1
        x = block(x)
    logits = head_submodel(x)

    streaming_model = Model(
        inputs=[frame_in] + cache_in_keras,
        outputs=[logits] + cache_outs,
        name="c1_tsmmbv2_streaming",
    )

    cfg = dict(
        n_classes=n_classes, T=T, input_hw=input_hw, in_channels=in_channels,
        width_mult=width_mult, n_blocks=n_blocks, shift_div=shift_div,
        stem_ch=stem_ch, last_ch=last_ch, flat_spec=flat_spec,
        n_tsm_sites=len(tsm_sites),
    )
    return {
        "cfg": cfg,
        "clip_model": clip_model,
        "streaming_model": streaming_model,
        "tsm_sites": tsm_sites,
        "blocks": blocks,
        "head": head_submodel,
        "stem": stem_submodel,
    }


# ---- Smoke test ----------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    tf.keras.utils.set_random_seed(0)

    cases = [
        dict(width_mult=0.25, n_blocks=17, expect_min=160_000, expect_max=350_000),  # C1a
        dict(width_mult=0.50, n_blocks=17, expect_min=600_000, expect_max=1_200_000),  # C1b
        dict(width_mult=0.25, n_blocks=10, expect_min=70_000, expect_max=220_000),   # C1c
    ]
    for case in cases:
        wm = case["width_mult"]; nb = case["n_blocks"]
        d = build_tsm_mbv2(width_mult=wm, n_blocks=nb)
        n_params = int(sum(int(np.prod(v.shape)) for v in d["clip_model"].trainable_weights))
        print(f"width_mult={wm}, n_blocks={nb}: params={n_params:,}  "
              f"tsm_sites={d['cfg']['n_tsm_sites']}  "
              f"final spatial={d['blocks'][-1].output_shape[1]}x{d['blocks'][-1].output_shape[2]}, "
              f"channels={d['blocks'][-1].output_shape[-1]}")
        # Smoke forward (clip)
        x = tf.random.normal((1, 8, 64, 64, 1))
        y = d["clip_model"](x, training=False)
        assert y.shape.as_list() == [1, 8, 27]
        # Smoke forward (streaming)
        f = tf.zeros((1, 64, 64, 1))
        caches = []
        for site in d["tsm_sites"]:
            # find spatial dim from streaming model input spec
            cache_input = [inp for inp in d["streaming_model"].inputs
                           if inp.name.startswith(f"cache_{site.name}_in")][0]
            caches.append(tf.zeros(cache_input.shape.as_list()))
        out_streaming = d["streaming_model"]([f] + caches, training=False)
        assert out_streaming[0].shape.as_list() == [1, 27]
        # Check params in expected range
        assert case["expect_min"] <= n_params <= case["expect_max"], \
            f"params {n_params} not in [{case['expect_min']}, {case['expect_max']}]"
    print("\nALL C1 SMOKE CHECKS PASSED")
