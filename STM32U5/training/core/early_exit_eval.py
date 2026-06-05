"""Day 12: early-exit strategy evaluation on INT8 TFLite clip models.

Strategies (Plan Day 12 Tier 1):
    * S1 (max-softmax):  exit at frame t when max(softmax(logits[t])) > thresh_s1
    * S3 (entropy):      exit at frame t when entropy(softmax(logits[t])) < thresh_s3

Inputs:
    - INT8 TFLite clip model: (1, T, H, W, 1) -> (1, T, K) logits
    - Jester val split (14787 samples; uses same crop/resize as eval_int8.py)

Pipeline (per sample):
    1. Run INT8 inference once -> logits[T, K]
    2. For each t in [0, T-1], compute softmax & decide exit
    3. Record (exit_frame, prediction, ground_truth)

Output (per strategy, per threshold):
    - n_correct, n_total, accuracy
    - mean observation ratio = mean((exit_frame+1) / T)
    - latency (ms): mean_obs_ratio * lat_per_frame  (Plan formula)

Default threshold grids cover the full quality/latency tradeoff curve.
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


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _entropy(p: np.ndarray) -> np.ndarray:
    """Entropy in nats along last axis. Adds tiny eps to avoid log(0)."""
    return -np.sum(p * np.log(p + 1e-12), axis=-1)


def _iter_val(T: int, img_size: int, max_samples: int):
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


def collect_logits(tflite_path: Path, T: int, img_size: int, max_samples: int):
    """Run INT8 inference once over val split; return (logits[N,T,K], labels[N])."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    in_is_int8 = inp["dtype"] == np.int8
    out_is_int8 = out["dtype"] == np.int8
    K = int(out["shape"][-1])

    print(f"  TFLite: in={inp['dtype'].__name__}  out={out['dtype'].__name__}  K={K}",
          flush=True)

    all_logits = []
    all_labels = []
    t0 = time.time()
    for i, (clip, label) in enumerate(_iter_val(T, img_size, max_samples)):
        x = clip[np.newaxis, ...].astype(np.float32)
        if in_is_int8:
            xq = np.round(x / in_scale + in_zp).clip(-128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], xq)
        else:
            interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])
        if out_is_int8:
            y_fp = (y.astype(np.float32) - out_zp) * out_scale
        else:
            y_fp = y.astype(np.float32)
        all_logits.append(y_fp[0])
        all_labels.append(int(label))
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}] elapsed={elapsed:.1f}s  rate={(i+1)/elapsed:.1f} samp/s",
                  flush=True)
    logits = np.stack(all_logits, axis=0)
    labels = np.asarray(all_labels, dtype=np.int32)
    print(f"  collected logits={logits.shape}  labels={labels.shape}  "
          f"in {time.time()-t0:.1f}s", flush=True)
    return logits, labels


def evaluate_strategy(strategy: str, logits: np.ndarray, labels: np.ndarray,
                      threshold: float, T: int, min_exit_frame: int = 0) -> dict:
    """Apply early-exit policy and report metrics.

    For each sample, find the smallest t s.t. policy(softmax(logits[t])) is met,
    subject to t >= min_exit_frame (floor — prevents exits at unreliable early
    frames). If no t meets the policy, exit at T-1 (full observation).

    Returns: {accuracy, mean_obs_ratio, exit_distribution[T]}
    """
    probs = _softmax(logits)            # (N, T, K)
    max_p = np.max(probs, axis=-1)      # (N, T)
    if strategy == "S1":
        meets = max_p > threshold       # (N, T) bool
    elif strategy == "S3":
        entr = _entropy(probs)          # (N, T)
        meets = entr < threshold
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    # Apply min_exit_frame floor: zero out meets[t] for t < min_exit_frame
    if min_exit_frame > 0:
        meets = meets.copy()
        meets[:, :min_exit_frame] = False

    # First-frame-true index along T axis. If no frame meets, default to T-1.
    any_meets = np.any(meets, axis=-1)
    first_meets = np.argmax(meets, axis=-1)
    exit_frame = np.where(any_meets, first_meets, T - 1)

    # Prediction = argmax at the exit frame
    preds = probs[np.arange(len(probs)), exit_frame].argmax(axis=-1)
    n_correct = int((preds == labels).sum())
    n_total = len(labels)
    accuracy = n_correct / max(n_total, 1)

    obs_ratio = (exit_frame + 1) / T
    mean_obs = float(obs_ratio.mean())

    # Exit-frame histogram
    exit_hist = np.bincount(exit_frame, minlength=T).tolist()

    return {
        "strategy": strategy,
        "threshold": threshold,
        "min_exit_frame": min_exit_frame,
        "accuracy": accuracy,
        "mean_obs_ratio": mean_obs,
        "n_correct": n_correct,
        "n_total": n_total,
        "exit_hist": exit_hist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, help="INT8 clip TFLite")
    ap.add_argument("--variant", required=True, help="C1f / C1j / ...")
    ap.add_argument("--tag", default="ptq", help="ptq / qat (for reporting)")
    ap.add_argument("--lat_per_frame_ms", type=float, default=0.0,
                    help="Per-frame latency from stedgeai. If >0, report total ms.")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="0 = full val (14787); else cap")
    ap.add_argument("--s1_thresholds", type=float, nargs="+",
                    default=[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99])
    ap.add_argument("--s3_thresholds", type=float, nargs="+",
                    default=[0.10, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00, 2.50, 3.00])
    ap.add_argument("--min_exit_frames", type=int, nargs="+", default=[0],
                    help="Floor(s): only allow exits at frame >= min_exit_frame. "
                         "Sweep multiple floors in one run. 0 = no floor.")
    ap.add_argument("--save_logits", default=None,
                    help="If set, dump collected logits to this .npz file for "
                         "later reuse via --logits_from")
    ap.add_argument("--logits_from", default=None,
                    help="Load logits + labels from .npz instead of TFLite inference. "
                         "Use when re-running threshold sweep on already-collected logits.")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    tf.get_logger().setLevel("ERROR")

    # Spec lookup for T / img_size
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

    print(f"[early-exit eval] {args.variant}/{args.tag}  tflite={args.tflite}")
    print(f"  T={T}  img={img_size}  max_samples={args.max_samples or 'full val'}")

    if args.logits_from:
        print(f"  loading logits from {args.logits_from}")
        data = np.load(args.logits_from)
        logits = data["logits"]
        labels = data["labels"]
        print(f"  loaded logits={logits.shape}  labels={labels.shape}")
    else:
        logits, labels = collect_logits(Path(args.tflite), T=T, img_size=img_size,
                                        max_samples=args.max_samples)
        if args.save_logits:
            np.savez(args.save_logits, logits=logits, labels=labels)
            print(f"  saved logits to {args.save_logits}")

    results = []
    # Baseline: no early exit, always observe full T (obs=1.0, accuracy at T-1)
    base_preds = _softmax(logits[:, -1, :]).argmax(axis=-1)
    base_acc = float((base_preds == labels).mean())
    print(f"\n  baseline (obs=1.0, no exit): acc = {base_acc:.4f}")
    results.append({
        "strategy": "BASELINE_NOEXIT",
        "threshold": None,
        "accuracy": base_acc,
        "mean_obs_ratio": 1.0,
        "n_correct": int((base_preds == labels).sum()),
        "n_total": len(labels),
        "exit_hist": [0] * (T - 1) + [len(labels)],
    })

    for mf in args.min_exit_frames:
        print(f"\n  ---- min_exit_frame = {mf} ----")
        print("  S1 (max-softmax > thresh):  thresh   acc    obs    lat(ms)")
        for th in args.s1_thresholds:
            r = evaluate_strategy("S1", logits, labels, th, T, min_exit_frame=mf)
            if args.lat_per_frame_ms > 0:
                r["latency_ms"] = round(r["mean_obs_ratio"] * args.lat_per_frame_ms * T, 2)
            results.append(r)
            lat_str = f"{r.get('latency_ms', 0):.1f}" if args.lat_per_frame_ms > 0 else "  -"
            print(f"    S1  mf={mf}  thresh={th:.3f}  acc={r['accuracy']:.4f}  "
                  f"obs={r['mean_obs_ratio']:.3f}  lat={lat_str}")

        print("  S3 (entropy < thresh):     thresh   acc    obs    lat(ms)")
        for th in args.s3_thresholds:
            r = evaluate_strategy("S3", logits, labels, th, T, min_exit_frame=mf)
            if args.lat_per_frame_ms > 0:
                r["latency_ms"] = round(r["mean_obs_ratio"] * args.lat_per_frame_ms * T, 2)
            results.append(r)
            lat_str = f"{r.get('latency_ms', 0):.1f}" if args.lat_per_frame_ms > 0 else "  -"
            print(f"    S3  mf={mf}  thresh={th:.3f}  acc={r['accuracy']:.4f}  "
                  f"obs={r['mean_obs_ratio']:.3f}  lat={lat_str}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "variant": args.variant, "tag": args.tag,
            "tflite": str(args.tflite), "T": T, "img_size": img_size,
            "lat_per_frame_ms": args.lat_per_frame_ms,
            "max_samples": args.max_samples,
            "baseline_acc": base_acc,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
