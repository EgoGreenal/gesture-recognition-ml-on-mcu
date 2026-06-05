"""Day 10: Per-variant analysis -- streaming-form export + stedgeai analyze + comparison CSV.

For each trained variant (V0..V6):
  1. Locate the latest run's best.weights.h5 in MODELS_ROOT/student_{name}/{job_id}/
  2. Build both clip + streaming-form Keras models (shared weights), load checkpoint
  3. Export streaming-form FP32 TFLite (reference, not used for deploy)
  4. Export streaming-form INT8 TFLite (PTQ; this is what U5 will run)
       - representative_dataset uses real Pipeline-L val data (32 samples)
       - cache_in tensors are sourced from a fresh streaming roll-out at t=0..3
         (zero on t=0, then real activations on later steps) so the calibrator
         sees the actual cache distribution, not random noise
  5. Run `stedgeai analyze --target stm32` on the INT8 TFLite to obtain:
       - real MACs / single frame
       - real RAM (activations) / ROM (weights)
       - confirm multi-I/O is still accepted with the trained scales
  6. Compute estimated U5 latency:
       latency_ms_per_clip = (MACs/frame * T) / 300e6 * 1000 + (T) * 5
       latency_ms_per_frame_avg = latency_ms_per_clip / T
  7. Append per-variant row to student_comparison_full.csv (final Day 10 table)
  8. Pareto filter:
       - hard constraint 1: SRAM peak <= 600 KB
       - hard constraint 2: estimated clip latency <= 200ms (5ms * 8 = 40ms ai_run overhead + compute)
       - rank by val_obs_1.00 among survivors; print top-3 recommended for QAT

The "INT8 PTQ val accuracy" check is **deferred to Day 11** (QAT step has its
own PTQ baseline) -- Day 10's job is variant ranking by FP32 acc + capability.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data_loader_tf import build_tf_dataset
from jester_common import parse_split
from student_model import VARIANTS, build_variant
from tsm_streaming_layer import TSMStreamingShift, causal_tsm_clip


# ---- TFLite export helpers --------------------------------------------------

def _build_streaming_representative_data(cfg, val_clips: list[np.ndarray], n_samples: int = 32):
    """For each calibration sample, run streaming model forward in pure Python
    using REAL data clip frames + the model's own previous-step cache outputs.

    This guarantees the calibrator sees the actual cache distribution rather
    than zeros / random noise.

    Returns a list of dicts (one per calibration step) keyed by input name —
    matches the `representative_dataset_factory` format from tsm_probe.py.
    """
    raise NotImplementedError("Per-clip rollout is generated lazily inside the factory below.")


def _representative_dataset_factory(
    cfg, blocks, head, val_clips: list[np.ndarray], input_names: list[str], n_samples: int
):
    """Yield {input_name: ndarray} for each calibration step.

    For each of `n_samples` val clips, we roll the streaming model forward T
    times. At each timestep we yield ONE sample (frame + caches at that step).
    Cache values come from the model's own previous-step outputs (real
    distribution, not zeros). Total yielded samples = n_samples * T.
    """
    # Build a fresh streaming model and load shared weights via the passed
    # blocks/head — but to compute cache_outs in Python we don't even need a
    # second model; we just call the same blocks + TSMStreamingShift layers.
    n_blocks = len(blocks)
    n_div = cfg.n_div
    streaming_shifts = [
        TSMStreamingShift(n_div=n_div, name=f"calib_tsm_{i}")
        for i in range(n_blocks - 1)
    ]
    # Pre-compute spatial sizes after each block (needed for cache zero init)
    spatial = []
    H_cur, W_cur = cfg.H, cfg.W
    for stride in cfg.strides:
        H_cur = (H_cur + stride - 1) // stride
        W_cur = (W_cur + stride - 1) // stride
        spatial.append((H_cur, W_cur))

    def _gen():
        rng = np.random.default_rng(0)
        for clip in val_clips[:n_samples]:
            # clip shape: (T, H, W, 1) float32
            caches = []
            for i in range(n_blocks - 1):
                Hi, Wi = spatial[i]
                n_shift = cfg.channels[i] // n_div
                caches.append(np.zeros((1, Hi, Wi, n_shift), dtype=np.float32))

            T = clip.shape[0]
            for t in range(T):
                frame = clip[t:t+1].astype(np.float32)   # (1, H, W, 1)
                # Build sample dict for this step
                sample = {}
                # input ordering by name — same as tsm_probe.py's matching
                for n in input_names:
                    if "frame" in n:
                        sample[n] = frame
                    elif "cache_" in n:
                        # name is like "serving_default_cache_2_in:0"; find idx
                        idx = int(n.split("cache_")[1].split("_")[0])
                        sample[n] = caches[idx]
                    else:
                        raise KeyError(f"unexpected calib input name: {n}")
                yield sample
                # Roll streaming model forward to update caches for next step
                x = tf.constant(frame)
                new_caches = []
                for i, block in enumerate(blocks):
                    x = block(x, training=False)
                    if i < n_blocks - 1:
                        x, c_out = streaming_shifts[i]([x, tf.constant(caches[i])])
                        new_caches.append(c_out.numpy())
                caches = new_caches
    return _gen


def export_streaming_int8_tflite(
    cfg, blocks, head, out_path: Path, saved_model_dir: Path, val_clips: list[np.ndarray],
    n_calib: int = 8,
) -> Path:
    """Build streaming-form Keras model, save as SavedModel, convert to INT8 TFLite."""
    from student_model import build_streaming_model
    stream_model = build_streaming_model(cfg, blocks, head)

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    stream_model.save(str(saved_model_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    sm = tf.saved_model.load(str(saved_model_dir))
    sig = sm.signatures["serving_default"]
    input_names = list(sig.structured_input_signature[1].keys())

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_factory(
        cfg, blocks, head, val_clips, input_names, n_calib,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)
    return out_path


# ---- stedgeai analyze wrapper (copies pattern from tsm_probe.py) ------------

def run_stedgeai_analyze(tflite_path: Path, out_dir: Path, name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "stedgeai", "analyze",
        "--model", str(tflite_path),
        "--target", "stm32",
        "--name", name,
        "--workspace", str(out_dir / "workspace"),
        "--output", str(out_dir / "output"),
        "--verbosity", "2",
    ]
    print(f"[analyze] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    log_text = result.stdout + "\n" + result.stderr
    (out_dir / "analyze_log.txt").write_text(log_text)

    import re
    def _find_int(pattern: str):
        m = re.search(pattern, log_text, re.IGNORECASE)
        return int(m.group(1).replace(",", "")) if m else None

    return {
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 2),
        "macc": _find_int(r"(?:total\s+)?(?:nb\s+of\s+)?macc[\s:]+([0-9,]+)"),
        "ram_bytes": _find_int(r"ram[^\d]+([0-9,]+)\s*(?:bytes|b)"),
        "rom_bytes": _find_int(r"rom[^\d]+([0-9,]+)\s*(?:bytes|b)"),
        "weights_bytes": _find_int(r"weights[^\d]+([0-9,]+)\s*(?:bytes|b)"),
        "log_path": str(out_dir / "analyze_log.txt"),
    }


# ---- main per-variant pipeline ---------------------------------------------

def find_latest_ckpt(models_root: Path, variant_name: str) -> tuple[Path, dict] | None:
    """Return (weights_path, meta) for the most recent run of `variant_name`.

    Prefers `best.weights.h5` (saved when per-epoch val ran). Falls back to
    `last.weights.h5` if no `best` exists (Day 9 train_student.py with
    val_every=0 doesn't produce best.h5).
    """
    var_dir = models_root / f"student_{variant_name}"
    if not var_dir.exists():
        return None
    # Look at all runs that have at least last.weights.h5
    candidates = [p for p in var_dir.iterdir()
                  if p.is_dir() and (p / "last.weights.h5").exists()]
    if not candidates:
        return None
    runs = sorted(candidates, key=lambda p: p.stat().st_mtime)
    latest = runs[-1]
    ckpt = latest / "best.weights.h5"
    if not ckpt.exists():
        ckpt = latest / "last.weights.h5"
    cfg = json.loads((latest / "config.json").read_text()) if (latest / "config.json").exists() else {}
    metrics = json.loads((latest / "metrics.json").read_text()) if (latest / "metrics.json").exists() else {}
    # If eval_students.py already wrote eval_full.json, splice the val numbers in
    eval_json = latest / "eval_full.json"
    if eval_json.exists():
        ef = json.loads(eval_json.read_text())
        metrics["best_val_obs_1.00"] = ef.get("metrics", {}).get("val_obs_1.00",
                                       metrics.get("best_val_obs_1.00", 0.0))
        metrics["eval_full"] = ef
    return ckpt, {"config": cfg, "metrics": metrics, "run_dir": str(latest)}


def analyze_one_variant(name: str, models_root: Path, n_calib: int = 4) -> dict:
    cfg = VARIANTS[name]
    found = find_latest_ckpt(models_root, name)
    if found is None:
        print(f"[skip] no checkpoint for {name}")
        return {"variant": name, "status": "no_ckpt"}
    ckpt_path, meta = found
    print(f"\n=== Analyzing {name}  ckpt={ckpt_path}")

    # Build models and load weights
    bundle = build_variant(cfg)
    clip_model = bundle["clip_model"]
    blocks = bundle["blocks"]
    head = bundle["head"]
    clip_model.load_weights(str(ckpt_path))
    print(f"  loaded weights")

    # Load some val clips for calibration
    val_ds = build_tf_dataset(
        split="validation", batch_size=1, T=cfg.T, img_size=cfg.H,
        training=False, num_shards=1,
    )
    val_clips = []
    for clip, _label in val_ds.take(n_calib):
        # clip shape (1, T, H, W, 1) — strip batch
        val_clips.append(clip.numpy()[0])
    print(f"  collected {len(val_clips)} calib clips")

    # Export streaming-form INT8 TFLite
    out_root = models_root / f"student_{name}" / "deploy"
    out_root.mkdir(parents=True, exist_ok=True)
    int8_path = out_root / f"student_{name}_streaming_int8.tflite"
    sm_dir = out_root / "saved_model_stream"
    export_streaming_int8_tflite(cfg, blocks, head, int8_path, sm_dir, val_clips, n_calib=n_calib)
    int8_size_kb = int8_path.stat().st_size / 1024
    print(f"  INT8 streaming TFLite: {int8_path}  ({int8_size_kb:.1f} KB)")

    # stedgeai analyze
    sai_dir = out_root / "stedgeai"
    sai = run_stedgeai_analyze(int8_path, sai_dir, name=f"student_{name}")
    print(f"  stedgeai: macc={sai.get('macc')} ram={sai.get('ram_bytes')} "
          f"rom={sai.get('rom_bytes')} weights={sai.get('weights_bytes')} "
          f"returncode={sai['returncode']}")

    # Latency estimate using Plan Section 5 formula
    macs_per_frame = sai.get("macc") or 0
    macs_per_clip = macs_per_frame * cfg.T  # ai_run per frame, but stedgeai is per-call
    # Note: stedgeai reports MACs for ONE invocation of the streaming model,
    # which corresponds to one frame. clip-level = macs_per_frame * T.
    latency_compute_ms = macs_per_clip / 300e6 * 1000
    latency_overhead_ms = cfg.T * 5.0   # one ai_run per frame, 5ms overhead each
    latency_clip_ms = latency_compute_ms + latency_overhead_ms
    latency_per_frame_ms = latency_clip_ms / cfg.T

    # FP32 val accuracy: prefer eval_full.json (post-training full val) if present
    best_val_obs_1 = float(meta["metrics"].get("best_val_obs_1.00", 0.0))
    eval_full = meta["metrics"].get("eval_full", {}).get("metrics", {})
    history = meta["metrics"].get("history", [])
    last = history[-1] if history else {}

    n_params = int(meta["config"].get("n_params", 0))
    n_cache = max(len(cfg.channels) - 1, 0)

    return {
        "variant": name,
        "status": "ok",
        "n_params": n_params,
        "channels": list(cfg.channels),
        "conv_type": cfg.conv_type,
        "n_div": cfg.n_div,
        "T": cfg.T, "H": cfg.H, "W": cfg.W,
        "n_blocks": len(cfg.channels),
        "n_cache": n_cache,
        "best_val_obs_1.00": round(eval_full.get("val_obs_1.00", best_val_obs_1), 4),
        "last_val_obs_0.75": round(eval_full.get("val_obs_0.75", last.get("obs_0.75", 0.0)), 4),
        "last_val_obs_0.50": round(eval_full.get("val_obs_0.50", last.get("obs_0.50", 0.0)), 4),
        "last_val_obs_0.25": round(eval_full.get("val_obs_0.25", last.get("obs_0.25", 0.0)), 4),
        "macs_per_frame": macs_per_frame,
        "macs_per_clip": macs_per_clip,
        "ram_bytes": sai.get("ram_bytes"),
        "rom_bytes": sai.get("rom_bytes"),
        "weights_bytes": sai.get("weights_bytes"),
        "int8_size_kb": round(int8_size_kb, 1),
        "latency_compute_ms_per_clip": round(latency_compute_ms, 1),
        "latency_overhead_ms_per_clip": round(latency_overhead_ms, 1),
        "latency_clip_ms": round(latency_clip_ms, 1),
        "latency_per_frame_ms": round(latency_per_frame_ms, 2),
        "ckpt": str(ckpt_path),
        "stedgeai_returncode": sai["returncode"],
    }


def pareto_filter(rows: list[dict], sram_budget_bytes: int, latency_budget_ms: float) -> list[dict]:
    survivors = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        ram = r.get("ram_bytes") or 0
        lat = r.get("latency_clip_ms") or 1e9
        if ram <= sram_budget_bytes and lat <= latency_budget_ms:
            survivors.append(r)
    survivors.sort(key=lambda r: -r["best_val_obs_1.00"])
    return survivors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    ap.add_argument("--models_root", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--n_calib", type=int, default=4,
                    help="num val clips used for streaming-form INT8 calibration "
                         "(each yields T sample steps).")
    ap.add_argument("--sram_budget_kb", type=int, default=600,
                    help="hard ceiling for activations RAM (Plan Section 5).")
    ap.add_argument("--latency_budget_ms", type=float, default=200.0,
                    help="hard ceiling for estimated U5 clip latency.")
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT / "models" / "student_comparison_full.csv"))
    args = ap.parse_args()
    models_root = Path(args.models_root)

    tf.get_logger().setLevel("ERROR")

    rows = []
    for name in args.variants:
        try:
            r = analyze_one_variant(name, models_root, n_calib=args.n_calib)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            r = {"variant": name, "status": "error", "error": str(e)}
        rows.append(r)

    # ---- Write CSV ----
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "variant", "status", "n_params", "n_blocks", "channels", "conv_type", "n_div",
        "H", "W", "T",
        "best_val_obs_1.00", "last_val_obs_0.75", "last_val_obs_0.50", "last_val_obs_0.25",
        "macs_per_frame", "macs_per_clip",
        "ram_bytes", "rom_bytes", "weights_bytes", "int8_size_kb",
        "latency_compute_ms_per_clip", "latency_overhead_ms_per_clip",
        "latency_clip_ms", "latency_per_frame_ms",
        "ckpt",
    ]
    lines = [",".join(header)]
    for r in rows:
        row = []
        for h in header:
            v = r.get(h, "")
            if isinstance(v, list):
                v = "-".join(map(str, v))
            row.append(str(v).replace(",", ";"))
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {csv_path}")

    # ---- Print summary + top-3 ----
    print("\n" + "=" * 110)
    print(f"{'V':<4}{'params':>9}{'acc1.0':>8}{'macs/f':>10}{'macs/clip':>12}"
          f"{'ram_KB':>8}{'lat_ms':>9}{'status':>10}")
    print("-" * 110)
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['variant']:<4}{'-':>9}{'-':>8}{'-':>10}{'-':>12}{'-':>8}{'-':>9}{r.get('status', '?'):>10}")
            continue
        print(
            f"{r['variant']:<4}{r['n_params']:>9,}"
            f"{r['best_val_obs_1.00']:>8.4f}"
            f"{r['macs_per_frame']:>10,}"
            f"{r['macs_per_clip']:>12,}"
            f"{(r['ram_bytes'] or 0)/1024:>8.1f}"
            f"{r['latency_clip_ms']:>9.1f}"
            f"{r['status']:>10}"
        )
    print("=" * 110)
    sram_b = args.sram_budget_kb * 1024
    survivors = pareto_filter(rows, sram_b, args.latency_budget_ms)
    print(f"\nPareto filter: ram <= {args.sram_budget_kb} KB AND latency <= {args.latency_budget_ms} ms")
    print(f"Survivors (sorted by FP32 val_obs_1.00):")
    for r in survivors:
        print(f"  {r['variant']}  acc1.0={r['best_val_obs_1.00']:.4f}  "
              f"lat={r['latency_clip_ms']:.0f}ms  ram={(r['ram_bytes'] or 0)/1024:.0f}KB")
    print(f"\n>>> Recommended top-3 for Day 11 QAT: {[r['variant'] for r in survivors[:3]]}")

    # Persist machine-readable summary
    summary_path = csv_path.parent / "analyze_summary.json"
    summary_path.write_text(json.dumps({
        "rows": rows, "survivors": [r["variant"] for r in survivors],
        "top_3": [r["variant"] for r in survivors[:3]],
        "sram_budget_kb": args.sram_budget_kb,
        "latency_budget_ms": args.latency_budget_ms,
    }, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
