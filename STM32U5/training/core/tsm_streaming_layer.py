"""Day 8: TSM streaming shift layer (TF/Keras), cache-as-input/output interface.

This is the **single most important piece** for Day 13 deployment: the
Streaming TSM block must (a) work in plain Keras for training, (b) convert
cleanly to TFLite + INT8 quantize, and (c) be representable in STM32CubeAI
12.0 with only standard ops (Slice / Concat) — no custom kernels.

Causal semantics (matching Day 6 TaTSMMobileNetV2.CausalTemporalShift):
  At each frame t, the first `C // n_div` channels carry the cache:
    output_t = concat( cache_in_t  ,  current_static_t , axis=channel )
    cache_out_t = current_shift_t   # i.e. cache becomes prev-frame's shift channels
  Where:
    input_t          = [B, H, W, C]  (current frame features)
    cache_in_t       = [B, H, W, C // n_div]  (from prev frame's call)
    current_shift_t  = input_t[..., :C // n_div]
    current_static_t = input_t[..., C // n_div:]

Equivalent to PyTorch:
    out[:, 1:, :fold] = x[:, :-1, :fold]   # (i.e. fold channels of t-1 placed at t)
    out[:, 0,  :fold] = 0                  # padded zero for t=0
    out[:, :,  fold:] = x[:, :,    fold:]  # remaining channels untouched

This file implements the **streaming-step** operator only (per-frame call,
cache as I/O tensor). The "clip" form (`[B, T, H, W, C]` shifted in time
along T) is **derived** for training by unrolling the streaming step in a
plain tf.scan or python for-loop — share weights and the same underlying
graph; the only difference is who owns the cache (Python loop vs C buffer).

TFLite Conversion notes:
  * Plain functional Keras layer with `Lambda` would be fine, BUT TFLite
    Converter prefers actual Keras `Layer` subclasses for clean export
  * `tf.split / tf.concat` are the only ops used — both `SLICE` + `CONCATENATION`
    in TFLite, both supported by STM32CubeAI 12.0 (verified Day 8)
  * No custom op needed -- this is the explicit reason we chose this design
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class TSMStreamingShift(layers.Layer):
    """Per-frame TSM streaming shift with cache-as-IO.

    Call: `(shifted_frame, cache_out) = layer((frame_in, cache_in))`

    Inputs:
        frame_in:  (B, H, W, C)          current-frame feature map
        cache_in:  (B, H, W, C // n_div) prev-frame's first `C // n_div` channels
                                          (zeros on t=0)

    Outputs:
        shifted_frame: (B, H, W, C)  causal-shifted version of frame_in
        cache_out:     (B, H, W, C // n_div)  this frame's shift channels,
                                              to be passed in as cache_in on t+1

    Note on shapes: all spatial sizes preserved (H, W identical I/O).
    Channel split point `n_shift = C // n_div` is a graph-time constant,
    not a runtime value -- this matters for TFLite Slice quantization scale.
    """

    def __init__(self, n_div: int = 8, name: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_div = int(n_div)

    def build(self, input_shape):
        # input_shape: [(B,H,W,C), (B,H,W,C//n_div)]
        if not (isinstance(input_shape, (list, tuple)) and len(input_shape) == 2):
            raise ValueError(
                f"TSMStreamingShift expects 2 inputs (frame, cache); got {input_shape}"
            )
        frame_shape, cache_shape = input_shape
        if frame_shape[-1] is None:
            raise ValueError("TSMStreamingShift requires static channel count")
        C = int(frame_shape[-1])
        if C % self.n_div != 0:
            raise ValueError(
                f"C={C} not divisible by n_div={self.n_div}; "
                "TSMStreamingShift needs static integer channel split"
            )
        self.C = C
        self.n_shift = C // self.n_div
        # Sanity: cache last dim must match n_shift
        if cache_shape[-1] is not None and int(cache_shape[-1]) != self.n_shift:
            raise ValueError(
                f"cache channels {cache_shape[-1]} != expected n_shift {self.n_shift}"
            )
        super().build(input_shape)

    def call(self, inputs):
        frame_in, cache_in = inputs
        # Split current frame into [shift_part, static_part]
        # Use tf.split with static integer sizes (gives clean Slice in TFLite).
        shift_part = frame_in[..., : self.n_shift]
        static_part = frame_in[..., self.n_shift :]
        # Output: replace shift_part with cache_in (prev-frame's shift channels)
        shifted = tf.concat([cache_in, static_part], axis=-1)
        # cache_out = this frame's shift_part (becomes cache_in at t+1)
        return shifted, shift_part

    def compute_output_shape(self, input_shape):
        frame_shape, cache_shape = input_shape
        return (frame_shape, cache_shape)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_div": self.n_div})
        return cfg


# ----------------------------------------------------------------------------
# Training-time helper: unroll the streaming step over T frames.
#
# Use this in the FP32 training graph: input is (B, T, H, W, C), output is
# (B, T, H, W, C) -- semantically identical to the PyTorch CausalTemporalShift
# (same graph as Day 6 TaTSMMobileNetV2.CausalTemporalShift but in TF).
# ----------------------------------------------------------------------------

def causal_tsm_clip(x: tf.Tensor, n_div: int = 8) -> tf.Tensor:
    """Apply causal TSM shift over the T axis of a clip tensor.

    Args:
        x:     (B, T, H, W, C)
        n_div: channel-fold divisor

    Returns:
        (B, T, H, W, C) with the first `C // n_div` channels at frame t
        replaced by the same channels at frame t-1 (zero-padded at t=0).
    """
    shape = x.shape
    if shape.rank != 5:
        raise ValueError(f"causal_tsm_clip expects rank-5 input, got {shape}")
    C = int(shape[-1])
    if C % n_div != 0:
        raise ValueError(f"C={C} not divisible by n_div={n_div}")
    n_shift = C // n_div
    # Split along channel
    shift_part = x[..., :n_shift]                     # (B, T, H, W, n_shift)
    static_part = x[..., n_shift:]                    # (B, T, H, W, C - n_shift)
    # Roll shift_part by +1 in T axis: shift_part_rolled[:, t] = shift_part[:, t-1]
    pad = tf.zeros_like(shift_part[:, :1])            # (B, 1, H, W, n_shift)
    shift_part_rolled = tf.concat(
        [pad, shift_part[:, :-1]], axis=1
    )                                                 # (B, T, H, W, n_shift)
    out = tf.concat([shift_part_rolled, static_part], axis=-1)   # (B, T, H, W, C)
    return out


# ----------------------------------------------------------------------------
# Sanity tests
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    tf.keras.utils.set_random_seed(0)

    B, T, H, W, C = 2, 8, 8, 8, 32
    n_div = 8
    n_shift = C // n_div

    # --- 1. clip form sanity (T-axis causal shift) ---
    x = tf.random.normal((B, T, H, W, C))
    y_clip = causal_tsm_clip(x, n_div=n_div)
    # First n_shift channels at t=0 must be zero
    assert tf.reduce_max(tf.abs(y_clip[:, 0, :, :, :n_shift])).numpy() == 0
    # First n_shift channels at t=k for k>=1 must equal x[:, k-1, :, :, :n_shift]
    diff = tf.reduce_max(tf.abs(y_clip[:, 1:, :, :, :n_shift] - x[:, :-1, :, :, :n_shift]))
    assert diff.numpy() == 0, f"clip causal shift diff = {diff}"
    # Last (C - n_shift) channels are unchanged
    diff2 = tf.reduce_max(tf.abs(y_clip[..., n_shift:] - x[..., n_shift:]))
    assert diff2.numpy() == 0
    print(f"causal_tsm_clip: PASSED on (B={B}, T={T}, H={H}, W={W}, C={C}, n_div={n_div})")

    # --- 2. streaming form sanity: unrolling streaming should match clip form ---
    layer = TSMStreamingShift(n_div=n_div)
    cache = tf.zeros((B, H, W, n_shift))
    outs = []
    for t in range(T):
        frame_t = x[:, t]                              # (B, H, W, C)
        shifted_t, cache_out = layer((frame_t, cache))
        outs.append(shifted_t)
        cache = cache_out                              # feed forward
    y_streaming = tf.stack(outs, axis=1)               # (B, T, H, W, C)
    diff = tf.reduce_max(tf.abs(y_streaming - y_clip))
    assert diff.numpy() < 1e-6, f"streaming vs clip diff = {diff.numpy()}"
    print(f"TSMStreamingShift unrolled vs causal_tsm_clip: PASSED  (max diff = {diff.numpy():.2e})")

    # --- 3. shapes ---
    print(f"streaming step output shapes: shifted={shifted_t.shape}, cache_out={cache_out.shape}")
    print("ALL TSM STREAMING SANITY CHECKS PASSED")
