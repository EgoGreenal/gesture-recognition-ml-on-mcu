"""Probe 2: try clone_model + quantize_apply workarounds for tfmot+Functional bug."""
from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import tensorflow_model_optimization as tfmot
from path_c_registry import build_clip_model
from tsm_streaming_layer import TSMStreamingShift

tf.get_logger().setLevel("ERROR")

clip_model, spec, stream, extras = build_clip_model("C1j")
print(f"clip type: {type(clip_model).__module__}.{type(clip_model).__name__}")
print(f"isinstance keras.Model: {isinstance(clip_model, tf.keras.Model)}")
print(f"isinstance tf.keras.models.Model: {isinstance(clip_model, tf.keras.models.Model)}")

# Attempt A: clone_model
print("\n--- A: clone_model(clip) then quantize_model ---")
try:
    cloned = tf.keras.models.clone_model(clip_model)
    print(f"  cloned type: {type(cloned).__name__}")
    q = tfmot.quantization.keras.quantize_model(cloned)
    print(f"  OK: quantized")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")

# Attempt B: quantize_apply directly (no annotation)
print("\n--- B: quantize_apply(clip) ---")
try:
    q = tfmot.quantization.keras.quantize_apply(clip_model)
    print(f"  OK")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")

# Attempt C: try on the streaming model with a clone
print("\n--- C: clone_model(streaming) then quantize_model ---")
try:
    cloned = tf.keras.models.clone_model(stream)
    print(f"  cloned type: {type(cloned).__name__}")
    with tfmot.quantization.keras.quantize_scope({"TSMStreamingShift": TSMStreamingShift}):
        q = tfmot.quantization.keras.quantize_model(cloned)
    print(f"  OK")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")

# Attempt D: build flat-inline clip model (test that approach)
print("\n--- D: test if tf.keras.Sequential of pure Conv2D works (sanity) ---")
try:
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 1)),
        tf.keras.layers.Conv2D(8, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(27),
    ])
    print(f"  sequential type: {type(m).__name__}")
    q = tfmot.quantization.keras.quantize_model(m)
    print(f"  OK: sanity-check Sequential model quantizes fine")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")

# Attempt E: try Functional but cast to keras.Model via Model(inputs, outputs)
print("\n--- E: tf.keras.Model(inputs, outputs) wrap then quantize ---")
try:
    wrapped = tf.keras.Model(clip_model.inputs, clip_model.outputs)
    print(f"  wrapped type: {type(wrapped).__name__}")
    q = tfmot.quantization.keras.quantize_model(wrapped)
    print(f"  OK")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")

print("\nALL DONE")
