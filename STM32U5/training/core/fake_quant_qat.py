"""Day 11: manual fake-quantize QAT (straight-through estimator) for C1 models.

Rationale: tfmot.quantization.keras.quantize_model() rejects nested-Functional
architectures with `to_annotate can only be a keras.Model instance` (tfmot 0.8.0
+ TF 2.15 regression). The C1 clip_model has TimeDistributed(Functional submodel)
nesting that triggers this. Rather than re-engineer the model into a flat graph,
we apply fake-quant ops directly at training time via straight-through estimator:

  forward:  w_q = clip(round(w / s) * s, -127s, 127s)
  backward: dL/dw = dL/dw_q                                   (STE)

This produces weights that, when converted via tf.lite.TFLiteConverter's INT8
PTQ path, suffer minimal accuracy drop because they were already trained against
the same rounding noise that PTQ introduces.

Symmetric per-channel weight quant (matching TFLite's INT8 default for Conv/Dense):
  scale = max(|W|, axis=non-out-channel) / 127
  W_q   = round(W / scale).clip(-127, 127) * scale

Activation quant is left to PTQ (representative dataset). This is "weight-only QAT",
which historically recovers 60-80% of the full QAT gain — adequate for our use case
since FP32 baselines are already 87%/86% (PTQ drop is expected to be small).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def _per_channel_fake_quant_weight(w: tf.Tensor, num_bits: int = 8) -> tf.Tensor:
    """Symmetric per-output-channel fake quant with STE.

    For Conv2D kernel shape (kH, kW, in_ch, out_ch): scale per out_ch.
    For Dense  kernel shape (in_dim, units):         scale per units.
    """
    qmax = (1 << (num_bits - 1)) - 1            # 127 for int8
    axes = list(range(w.shape.rank - 1))        # all but last
    abs_max = tf.reduce_max(tf.abs(w), axis=axes, keepdims=True) + 1e-9
    scale = abs_max / qmax
    w_int = tf.round(w / scale)
    w_int = tf.clip_by_value(w_int, -qmax, qmax)
    w_q = w_int * scale
    # STE: forward = w_q, backward = identity wrt w
    return w + tf.stop_gradient(w_q - w)


class FakeQuantConv2D(layers.Conv2D):
    """Conv2D variant that fake-quantizes its kernel on every forward pass.

    Use as a drop-in replacement when training; export via TFLite Converter
    INT8 PTQ at the end -- the converter will see weight ranges that have been
    "rounded into" by training and produce a cleaner INT8 model.
    """
    def __init__(self, *args, qat_bits: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_bits = int(qat_bits)

    def call(self, inputs):
        # Temporarily swap kernel with fake-quant version, only for this call
        orig = self.kernel
        try:
            self.kernel = _per_channel_fake_quant_weight(orig, self.qat_bits)
            out = super().call(inputs)
        finally:
            self.kernel = orig
        return out


class FakeQuantDense(layers.Dense):
    def __init__(self, *args, qat_bits: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_bits = int(qat_bits)

    def call(self, inputs):
        orig = self.kernel
        try:
            self.kernel = _per_channel_fake_quant_weight(orig, self.qat_bits)
            out = super().call(inputs)
        finally:
            self.kernel = orig
        return out


def install_fake_quant_on_model(model: tf.keras.Model, num_bits: int = 8) -> int:
    """Monkey-patch all Conv2D / DepthwiseConv2D / Dense layers in `model`
    (recursing into TimeDistributed and sub-models) so their `.call()`
    fake-quantizes the kernel each forward pass.

    Returns the number of layers patched.

    Side effects: each patched layer gets a `_orig_call` attribute (rollback
    hook). The patch is symmetric per-channel weight quant with STE; activations
    are NOT touched (they remain FP32 in graph; INT8 activation ranges are
    learned by the TFLite Converter's representative_dataset pass).
    """
    qmax = (1 << (num_bits - 1)) - 1
    patched = 0

    def _patch_one(lyr: tf.keras.layers.Layer):
        nonlocal patched
        # Recurse into wrappers and sub-models
        if isinstance(lyr, layers.TimeDistributed):
            _patch_one(lyr.layer)
            return
        if isinstance(lyr, tf.keras.Model):
            for sub in lyr.layers:
                _patch_one(sub)
            return
        # Conv2D / DepthwiseConv2D / Dense have a `.kernel` attribute (and
        # for DepthwiseConv2D, `.depthwise_kernel`).
        target_attr = None
        if isinstance(lyr, layers.DepthwiseConv2D):
            target_attr = "depthwise_kernel"
        elif isinstance(lyr, (layers.Conv2D, layers.Dense)):
            target_attr = "kernel"
        if target_attr is None:
            return
        if getattr(lyr, "_fake_quant_installed", False):
            return
        orig_call = lyr.call
        bits = num_bits

        def patched_call(inputs, *args, _orig_call=orig_call,
                         _attr=target_attr, _bits=bits, _lyr=lyr, **kwargs):
            orig = getattr(_lyr, _attr)
            try:
                fq = _per_channel_fake_quant_weight(orig, _bits)
                setattr(_lyr, _attr, fq)
                return _orig_call(inputs, *args, **kwargs)
            finally:
                setattr(_lyr, _attr, orig)

        lyr.call = patched_call
        lyr._fake_quant_installed = True
        lyr._orig_call = orig_call
        patched += 1

    for lyr in model.layers:
        _patch_one(lyr)
    return patched


def uninstall_fake_quant(model: tf.keras.Model) -> int:
    """Roll back `install_fake_quant_on_model` patches. Returns # rolled back."""
    rolled = 0

    def _unpatch(lyr):
        nonlocal rolled
        if isinstance(lyr, layers.TimeDistributed):
            _unpatch(lyr.layer)
            return
        if isinstance(lyr, tf.keras.Model):
            for sub in lyr.layers:
                _unpatch(sub)
            return
        if getattr(lyr, "_fake_quant_installed", False):
            lyr.call = lyr._orig_call
            lyr._fake_quant_installed = False
            del lyr._orig_call
            rolled += 1

    for lyr in model.layers:
        _unpatch(lyr)
    return rolled


# ---- Sanity ---------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/scripts")))
    from path_c_registry import build_clip_model

    tf.keras.utils.set_random_seed(0)
    clip_model, spec, _, _ = build_clip_model("C1j")
    n = install_fake_quant_on_model(clip_model)
    print(f"Patched {n} Conv2D/DepthwiseConv2D/Dense layers")
    x = tf.random.normal((1, 8, 64, 64, 1))
    y = clip_model(x, training=True)
    print(f"forward (training=True) output shape: {y.shape}  "
          f"mean={float(tf.reduce_mean(y)):.4f}  std={float(tf.math.reduce_std(y)):.4f}")
    y2 = clip_model(x, training=False)
    print(f"forward (training=False) output shape: {y2.shape}")
    # Diff between fake-quant-on vs original FP32: should be small but nonzero
    m2 = uninstall_fake_quant(clip_model)
    print(f"Rolled back {m2} layers")
    y3 = clip_model(x, training=False)
    diff = float(tf.reduce_max(tf.abs(y3 - y)))
    print(f"diff between fake-quant-on and FP32 logits: max={diff:.5f}")
    assert diff < 1.0, "fake-quant logits should be close to FP32 logits"
    print("FAKE-QUANT QAT SANITY PASSED")
