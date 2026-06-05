"""Day 11 probe: see what tfmot.quantize_model does on a C1 clip_model.

Goal: determine whether the standard tfmot path is viable for our TimeDistributed
+ Lambda-rich architecture, or whether we need either:
  (a) a flat-inlined clip rebuild, or
  (b) manual fake-quant fine-tune, or
  (c) skip QAT and accept PTQ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import tensorflow_model_optimization as tfmot
from path_c_registry import build_clip_model

tf.get_logger().setLevel("ERROR")


def try_quantize(variant: str):
    print(f"\n=== {variant} ===", flush=True)
    clip_model, spec, stream, _ = build_clip_model(variant)
    print(f"clip layers: {len(clip_model.layers)}; streaming: "
          f"{len(stream.layers) if stream else 'n/a'}; family={spec.family}")
    print("clip top-level layer types:")
    for lyr in clip_model.layers[:15]:
        print(f"  {type(lyr).__name__:<25}  name={lyr.name}")

    # Attempt 1: bare quantize_model on clip_model
    print("\nAttempt 1: tfmot.quantization.keras.quantize_model(clip_model)")
    try:
        q = tfmot.quantization.keras.quantize_model(clip_model)
        print(f"  OK: quantized model has {len(q.layers)} layers")
        return "clip_ok"
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")

    # Attempt 2: bare quantize_model on streaming_model
    if stream is not None:
        print("\nAttempt 2: tfmot.quantization.keras.quantize_model(streaming_model)")
        try:
            with tfmot.quantization.keras.quantize_scope():
                q = tfmot.quantization.keras.quantize_model(stream)
            print(f"  OK: quantized streaming model has {len(q.layers)} layers")
            return "stream_ok"
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")
    return "both_fail"


if __name__ == "__main__":
    for v in ["C1f", "C1j"]:
        try_quantize(v)
