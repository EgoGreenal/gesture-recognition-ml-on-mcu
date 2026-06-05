"""Day 11: export INT8 TFLite from a C1 FP32 ckpt (PTQ) or QAT ckpt.

Used by both:
    * PTQ baseline:  FP32 ckpt -> INT8 via representative dataset (no fine-tune)
    * QAT path:      QAT ckpt (fake-quant-aware) -> INT8 via representative
                     dataset (calib still needed for activation ranges that
                     the fake-quant ops did not absorb)

We export TWO graphs:
    * clip_int8.tflite      — (B,T,H,W,C) -> (B,T,K). For host-side INT8
                               accuracy evaluation (eval_int8.py reads this).
    * streaming_int8.tflite — multi-input (frame + N caches) -> (logits + N
                               caches). For stedgeai analyze (deployment graph).

Day 10 lesson reused: multi-input streaming-model INT8 PTQ often fails due to
TFLite Converter reordering input ports vs. our positional generator. We
attempt INT8 first; on failure we save FP32 with a logged warning. Graph-level
metrics (MACs/RAM via stedgeai) do not depend on calibration realism.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from jester_common import H5_PATH, parse_split, sample_segment_indices
from path_c_registry import build_clip_model


def _load_val_clips(n: int, T: int, img_size: int) -> np.ndarray:
    """Load N validation clips (T, H, W, 1) float32 in [-1, 1]."""
    items = parse_split("validation")[: n * 2]
    selected = items[:n]
    clips = []
    with h5py.File(str(H5_PATH), "r") as h5:
        for vid, _label in selected:
            ds = h5["clips"][str(vid)]
            n_frames = ds.shape[0]
            offsets = sample_segment_indices(n_frames, T, training=False)
            all_bytes = ds[:]
            jpegs = [bytes(all_bytes[int(o)]) for o in offsets.tolist()]
            first = Image.open(io.BytesIO(jpegs[0])).convert("L")
            w, h = first.size
            side = min(w, h); top = (h - side) // 2; left = (w - side) // 2
            frames = np.empty((T, img_size, img_size, 1), dtype=np.float32)
            for k, jpg in enumerate(jpegs):
                img = Image.open(io.BytesIO(jpg)).convert("L")
                img = img.crop((left, top, left + side, top + side)).resize(
                    (img_size, img_size), Image.BILINEAR
                )
                arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                frames[k, ..., 0] = arr
            clips.append(frames)
    return np.stack(clips, axis=0)  # (N, T, H, W, 1)


def export_clip_int8(clip_model: tf.keras.Model, val_clips: np.ndarray,
                     out_path: Path, n_calib: int = 16) -> tuple[Path, bool]:
    """Convert clip model (B,T,H,W,C)->(B,T,K) to INT8 TFLite.

    Single-input model — easiest case for the TFLite Converter.
    Returns (out_path, is_int8).
    """
    def gen():
        for i in range(min(n_calib, val_clips.shape[0])):
            yield [val_clips[i:i+1].astype(np.float32)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(clip_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_bytes = converter.convert()
        out_path.write_bytes(tflite_bytes)
        print(f"  [clip INT8] wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
        return out_path, True
    except Exception as e:
        print(f"  [clip INT8 FAIL] {type(e).__name__}: {str(e)[:200]}")
        # Fallback: FP32 clip TFLite (host-side eval still works, but no INT8 acc)
        converter = tf.lite.TFLiteConverter.from_keras_model(clip_model)
        tflite_bytes = converter.convert()
        fp32_path = out_path.with_suffix(".fp32.tflite")
        fp32_path.write_bytes(tflite_bytes)
        print(f"  [clip FP32 fallback] wrote {fp32_path}  ({fp32_path.stat().st_size/1024:.1f} KB)")
        return fp32_path, False


def export_streaming_int8(streaming_model: tf.keras.Model, val_clips: np.ndarray,
                          out_path: Path, n_calib: int = 16,
                          via_saved_model: bool = False,
                          saved_model_dir: Path | None = None) -> tuple[Path, bool]:
    """Convert streaming model to INT8 TFLite for deployment.

    Multi-input (frame + N caches). Two paths:
      (1) from_keras_model + positional generator -- Day 10 lesson: fails on
          multi-input due to TFLite Converter input reordering + calibrator
          channel checks
      (2) from_saved_model + dict-keyed generator (via_saved_model=True) --
          SavedModel exposes signatures with named inputs, so the generator
          dict-yields keyed by signature input name. This is the Day 10
          documented workaround that has not been tested yet (Day 13 prep).
    Both fall back to FP32 if INT8 conversion fails.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_dict_gen = bool(via_saved_model)

    if via_saved_model:
        if saved_model_dir is None:
            saved_model_dir = out_path.parent / "saved_model_streaming"
        saved_model_dir = Path(saved_model_dir)
        if saved_model_dir.exists():
            import shutil as _shutil
            _shutil.rmtree(saved_model_dir)
        # tf.saved_model.save writes a SavedModel directory; signatures default
        # to the model.call signature with positional input names.
        print(f"  [saved_model] writing {saved_model_dir}")
        streaming_model.export(str(saved_model_dir))
        # Re-load to discover the signature input names (these are what the
        # TFLite Converter will surface as dict keys for representative_dataset)
        loaded = tf.saved_model.load(str(saved_model_dir))
        sig = loaded.signatures["serving_default"]
        input_specs = list(sig.structured_input_signature[1].items())
        input_names = [name for name, _ in input_specs]
        input_shapes_by_name = {name: [d if d is not None else 1 for d in spec.shape]
                                for name, spec in input_specs}
        print(f"  [saved_model] signature inputs (dict keys): {input_names}")
    else:
        input_shapes_by_name = None
        input_names = None

    # Build generator. If using SavedModel path, yield dict keyed on signature
    # names; otherwise yield positional list matching streaming_model.inputs.
    if use_dict_gen:
        def gen():
            for i in range(min(n_calib, val_clips.shape[0])):
                clip = val_clips[i]                # (T, H, W, 1)
                T = clip.shape[0]
                for t in range(T):
                    sample = {}
                    for name in input_names:
                        shape = input_shapes_by_name[name]
                        # First port is the per-frame video input; rest are caches
                        # Heuristic: "frame" in name -> frame; else cache (zeros)
                        if "frame" in name.lower() or "input_1" in name.lower():
                            sample[name] = clip[t:t+1].astype(np.float32)
                        else:
                            sample[name] = np.zeros(shape, dtype=np.float32)
                    yield sample
    else:
        input_shapes = [[d if d is not None else 1 for d in inp.shape]
                        for inp in streaming_model.inputs]

        def gen():
            for i in range(min(n_calib, val_clips.shape[0])):
                clip = val_clips[i]                # (T, H, W, 1)
                T = clip.shape[0]
                for t in range(T):
                    sample_list = []
                    for j, shape in enumerate(input_shapes):
                        if j == 0:
                            sample_list.append(clip[t:t+1].astype(np.float32))
                        else:
                            sample_list.append(np.zeros(shape, dtype=np.float32))
                    yield sample_list

    try:
        if via_saved_model:
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(streaming_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_bytes = converter.convert()
        out_path.write_bytes(tflite_bytes)
        print(f"  [streaming INT8 via_saved_model={via_saved_model}] wrote {out_path}  "
              f"({out_path.stat().st_size/1024:.1f} KB)")
        return out_path, True
    except Exception as e:
        print(f"  [streaming INT8 FAIL via_saved_model={via_saved_model}] "
              f"{type(e).__name__}: {str(e)[:300]}")
        converter = tf.lite.TFLiteConverter.from_keras_model(streaming_model)
        tflite_bytes = converter.convert()
        fp32_path = out_path.with_suffix(".fp32.tflite")
        fp32_path.write_bytes(tflite_bytes)
        print(f"  [streaming FP32 fallback] wrote {fp32_path}  ({fp32_path.stat().st_size/1024:.1f} KB)")
        return fp32_path, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="C1f / C1j / ...")
    ap.add_argument("--ckpt", required=True,
                    help="Path to FP32 best.weights.h5 (PTQ) or QAT best.weights.h5")
    ap.add_argument("--tag", required=True, choices=["ptq", "qat"],
                    help="Suffix tag for output filenames")
    ap.add_argument("--out_dir", default=None,
                    help="Defaults to models/student_<V>/deploy")
    ap.add_argument("--n_calib", type=int, default=16)
    ap.add_argument("--export_clip", action="store_true",
                    help="Also export clip-form INT8 (for eval_int8.py)")
    ap.add_argument("--export_streaming", action="store_true",
                    help="Also export streaming-form INT8 (for stedgeai)")
    ap.add_argument("--via_saved_model", action="store_true",
                    help="Use SavedModel + dict-keyed representative dataset for "
                         "streaming export. Day 10 documented workaround for "
                         "multi-input INT8 PTQ failures with from_keras_model.")
    args = ap.parse_args()

    tf.get_logger().setLevel("ERROR")
    t0 = time.time()
    clip_model, spec, streaming_model, _ = build_clip_model(args.variant)
    clip_model.load_weights(args.ckpt)
    print(f"[{args.variant}/{args.tag}] loaded weights from {args.ckpt}")
    print(f"  family={spec.family}  T={spec.T}  img={spec.img_size}")

    val_clips = _load_val_clips(args.n_calib, spec.T, spec.img_size)
    print(f"  calib clips: {val_clips.shape}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "models" / f"student_{spec.name}" / "deploy"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (args.export_clip or args.export_streaming):
        # Default: export both
        args.export_clip = True
        args.export_streaming = True

    results = {}
    if args.export_clip:
        clip_out = out_dir / f"{args.variant}_{args.tag}_clip_int8.tflite"
        path, is_int8 = export_clip_int8(clip_model, val_clips, clip_out, args.n_calib)
        results["clip"] = {"path": str(path), "int8": is_int8,
                           "size_kb": round(path.stat().st_size / 1024, 1)}
    if args.export_streaming:
        if streaming_model is None:
            print("  [streaming] skipped — streaming_model is None for this variant")
        else:
            suffix = "_sm" if args.via_saved_model else ""
            stream_out = out_dir / f"{args.variant}_{args.tag}_streaming{suffix}_int8.tflite"
            path, is_int8 = export_streaming_int8(
                streaming_model, val_clips, stream_out, args.n_calib,
                via_saved_model=args.via_saved_model,
                saved_model_dir=(out_dir / f"saved_model_streaming_{args.variant}"
                                 if args.via_saved_model else None),
            )
            results["streaming"] = {"path": str(path), "int8": is_int8,
                                    "size_kb": round(path.stat().st_size / 1024, 1)}

    print(f"\nDONE in {time.time()-t0:.1f}s")
    print("results:")
    for k, v in results.items():
        kind = "INT8" if v["int8"] else "FP32"
        print(f"  {k}: {kind}  {v['size_kb']} KB  {v['path']}")


if __name__ == "__main__":
    main()
