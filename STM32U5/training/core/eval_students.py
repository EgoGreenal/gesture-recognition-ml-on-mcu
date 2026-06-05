"""Day 10: Standalone validation runner for trained student variants.

Loads a variant's best.weights.h5 (or last.weights.h5) and runs **full** Jester
validation in eager mode, computing top-1 accuracy at each observation ratio.

Use cases:
  * Day 10 final variant comparison (gold-standard val numbers)
  * If train_student.py skipped per-epoch val (--val_every 0), this is how the
    val acc gets computed before analyze_students.py builds the comparison table
  * Re-evaluating an older checkpoint without re-running the train loop

CLI:
    python eval_students.py --variants V0 V1 ... [--use_last]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data_loader_tf import build_tf_dataset
from student_model import VARIANTS, build_variant
from train_student import evaluate


def find_latest_run(models_root: Path, variant: str, prefer: str = "best") -> tuple[Path, dict] | None:
    """Return (weights_path, meta dict) for latest run.

    Prefer `prefer.weights.h5`; fall back to the other (Day 9 train with
    val_every=0 produces only last.weights.h5, no best).
    """
    var_dir = models_root / f"student_{variant}"
    if not var_dir.exists():
        return None
    fallback = "last" if prefer == "best" else "best"
    # accept any run that has either of the two checkpoints
    runs = [p for p in var_dir.iterdir() if p.is_dir() and (
        (p / f"{prefer}.weights.h5").exists() or (p / f"{fallback}.weights.h5").exists()
    )]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime)
    latest = runs[-1]
    meta = {}
    cj = latest / "config.json"
    mj = latest / "metrics.json"
    if cj.exists():
        meta["config"] = json.loads(cj.read_text())
    if mj.exists():
        meta["metrics"] = json.loads(mj.read_text())
    meta["run_dir"] = str(latest)
    ckpt = latest / f"{prefer}.weights.h5"
    if not ckpt.exists():
        ckpt = latest / f"{fallback}.weights.h5"
    return ckpt, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    ap.add_argument("--models_root", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_shards", type=int, default=2,
                    help="Val data loader shards. Day 9: keep modest to avoid HDF5 contention.")
    ap.add_argument("--use_last", action="store_true",
                    help="Use last.weights.h5 instead of best.weights.h5")
    ap.add_argument("--max_batches", type=int, default=0,
                    help="0 = full val (default); >0 caps for quick checks")
    args = ap.parse_args()

    tf.get_logger().setLevel("ERROR")
    prefer = "last" if args.use_last else "best"
    models_root = Path(args.models_root)

    rows = []
    for name in args.variants:
        found = find_latest_run(models_root, name, prefer=prefer)
        if found is None:
            print(f"[skip] no {prefer}.weights.h5 for {name}")
            rows.append({"variant": name, "status": "no_ckpt"})
            continue
        ckpt, meta = found
        cfg = VARIANTS[name]
        print(f"\n=== Eval {name}  ckpt={ckpt}")

        bundle = build_variant(cfg)
        clip_model = bundle["clip_model"]
        clip_model.load_weights(str(ckpt))

        val_ds = build_tf_dataset(
            split="validation",
            batch_size=args.batch_size,
            T=cfg.T,
            img_size=cfg.H,
            training=False,
            num_shards=args.num_shards,
            seed=42,
        )
        t0 = time.time()
        m = evaluate(clip_model, val_ds, cfg.obs_ratios, T_s=cfg.T,
                     max_batches=args.max_batches)
        del val_ds   # release HDF5 handles
        t_val = time.time() - t0
        print(f"  val_obs_1.00 = {m.get('obs_1.00', 0):.4f}  "
              f"val_obs_0.75 = {m.get('obs_0.75', 0):.4f}  "
              f"val_obs_0.50 = {m.get('obs_0.50', 0):.4f}  "
              f"val_obs_0.25 = {m.get('obs_0.25', 0):.4f}  "
              f"({m['n_val_samples']} samples in {t_val:.1f}s)")
        rows.append({
            "variant": name, "status": "ok", "ckpt": str(ckpt),
            "val_obs_1.00": round(m.get("obs_1.00", 0), 4),
            "val_obs_0.75": round(m.get("obs_0.75", 0), 4),
            "val_obs_0.50": round(m.get("obs_0.50", 0), 4),
            "val_obs_0.25": round(m.get("obs_0.25", 0), 4),
            "n_val_samples": m["n_val_samples"],
            "val_time_s": round(t_val, 1),
        })

        # Write a small JSON next to the checkpoint so analyze_students.py can pick it up
        eval_json = Path(meta["run_dir"]) / "eval_full.json"
        eval_json.write_text(json.dumps({
            "ckpt": str(ckpt), "variant": name, "metrics": rows[-1],
        }, indent=2))
        print(f"  wrote {eval_json}")

    # Final report
    print("\n=== Eval summary ===")
    print(f"{'V':<4}{'acc1.0':>9}{'acc0.75':>9}{'acc0.50':>9}{'acc0.25':>9}{'n':>8}{'t_s':>7}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['variant']:<4}  ({r.get('status', '?')})")
            continue
        print(
            f"{r['variant']:<4}{r['val_obs_1.00']:>9.4f}{r['val_obs_0.75']:>9.4f}"
            f"{r['val_obs_0.50']:>9.4f}{r['val_obs_0.25']:>9.4f}"
            f"{r['n_val_samples']:>8}{r['val_time_s']:>7.1f}"
        )


if __name__ == "__main__":
    main()
