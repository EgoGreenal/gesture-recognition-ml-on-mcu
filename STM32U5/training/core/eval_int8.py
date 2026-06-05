"""Day 11: evaluate INT8 TFLite top-1 accuracy on Jester val.

Works on clip-form INT8 TFLite (`<V>_<tag>_clip_int8.tflite`): single input
(1,T,H,W,C) -> (1,T,K). We score at obs_ratio=1.0 (last frame logits) which
is the same metric the FP32 training pipeline uses for `val_obs_1.00`, so the
numbers are directly comparable for measuring INT8 PTQ/QAT drop.

If the input file is FP32 (.fp32.tflite suffix), we still run the eval and
report the number — that's the "INT8 export failed, FP32 fallback" path.

Output: prints summary, writes a JSON with per-(variant,tag) acc.
"""
from __future__ import annotations

import argparse
import io
import json
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


def _iter_val(T: int, img_size: int, max_samples: int):
    """Yield (clip[T,H,W,1] float32 in [-1,1], label) for val split."""
    items = parse_split("validation")
    if max_samples and max_samples > 0:
        items = items[:max_samples]
    with h5py.File(str(H5_PATH), "r") as h5:
        for vid, label in items:
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
            yield frames, label


def eval_clip_tflite(tflite_path: Path, T: int, img_size: int,
                     max_samples: int = 0, obs_ratios=(1.0, 0.75, 0.5, 0.25)) -> dict:
    """Top-1 acc at multiple obs ratios on val split."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  input dtype={inp['dtype'].__name__}  shape={inp['shape']}  "
          f"scale={inp['quantization']}")
    print(f"  output dtype={out['dtype'].__name__}  shape={out['shape']}  "
          f"scale={out['quantization']}")

    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    in_is_int8 = inp["dtype"] == np.int8
    out_is_int8 = out["dtype"] == np.int8

    # Verify shape compatibility (1, T, H, W, 1)
    exp = (1, T, img_size, img_size, 1)
    if tuple(inp["shape"]) != exp:
        print(f"  WARN: expected shape {exp}, got {tuple(inp['shape'])}")

    fi = [max(int(round(T * r)) - 1, 0) for r in obs_ratios]
    n_correct = [0] * len(fi)
    n_total = 0
    t0 = time.time()
    for clip, label in _iter_val(T, img_size, max_samples):
        x = clip[np.newaxis, ...].astype(np.float32)  # (1,T,H,W,1)
        if in_is_int8:
            xq = np.round(x / in_scale + in_zp).clip(-128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], xq)
        else:
            interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])  # (1, T, K)
        if out_is_int8:
            y_fp = (y.astype(np.float32) - out_zp) * out_scale
        else:
            y_fp = y.astype(np.float32)
        for k, idx in enumerate(fi):
            pred = int(np.argmax(y_fp[0, idx]))
            if pred == int(label):
                n_correct[k] += 1
        n_total += 1
        if n_total % 500 == 0:
            elapsed = time.time() - t0
            cur_acc1 = n_correct[0] / n_total
            print(f"    [{n_total}] acc@1.0={cur_acc1:.4f}  "
                  f"throughput={n_total/elapsed:.1f} samp/s", flush=True)

    elapsed = time.time() - t0
    result = {f"obs_{r:.2f}": n_correct[k] / max(n_total, 1)
              for k, r in enumerate(obs_ratios)}
    result["n_val"] = n_total
    result["elapsed_s"] = round(elapsed, 1)
    result["throughput_samp_per_s"] = round(n_total / max(elapsed, 1e-6), 2)
    result["tflite_path"] = str(tflite_path)
    result["tflite_dtype"] = "int8" if in_is_int8 else "fp32"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, help="Path to clip-form TFLite")
    ap.add_argument("--variant", required=True, help="C1f / C1j (for spec lookup)")
    ap.add_argument("--tag", default="unknown", help="ptq / qat / fp32 (for reporting)")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="0 = full val (14787 samples); else cap")
    ap.add_argument("--out_json", default=None,
                    help="Append result to this JSON (otherwise stdout only)")
    args = ap.parse_args()

    tf.get_logger().setLevel("ERROR")

    # Spec lookup
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from path_c_registry import get_spec, C1_PRESETS
    spec = get_spec(args.variant)
    if spec.family == "C1":
        preset = C1_PRESETS[spec.name]
        T = spec.T
        img_size = preset["input_hw"]
    else:
        T = spec.T
        img_size = spec.img_size

    print(f"[{args.variant}/{args.tag}] eval {args.tflite}")
    print(f"  T={T}  img={img_size}  max_samples={args.max_samples or 'full val'}")
    result = eval_clip_tflite(Path(args.tflite), T=T, img_size=img_size,
                              max_samples=args.max_samples)
    result["variant"] = args.variant
    result["tag"] = args.tag

    print("\nINT8 eval result:")
    for k in ("obs_0.25", "obs_0.50", "obs_0.75", "obs_1.00"):
        print(f"  {k} = {result.get(k, 0):.4f}")
    print(f"  n_val={result['n_val']}  elapsed={result['elapsed_s']}s  "
          f"throughput={result['throughput_samp_per_s']} samp/s")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        else:
            existing = []
        existing.append(result)
        out_path.write_text(json.dumps(existing, indent=2))
        print(f"\nappended to {out_path}")


if __name__ == "__main__":
    main()
