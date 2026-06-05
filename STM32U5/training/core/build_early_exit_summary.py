"""Day 12: aggregate early-exit operating points + pick (variant, strategy, threshold)
for U5 deployment.

Reads:
    models/early_exit_<V>_ptq_v2.json   # has min_exit_frame sweep
    models/stedgeai_<V>_ptq.json        # for lat_per_frame

Output:
    models/early_exit_comparison.csv    # ALL operating points across (variant,
                                          strategy, threshold, min_exit_frame)
    models/early_exit_winners.json      # picked winners for U5

Winner picking rule (Plan Day 12 + Day 13 constraints):
    - acc >= baseline_acc - 1pp   (don't sacrifice >1pp of accuracy)
    - minimize  obs_ratio (= avg observation ratio → avg latency)
    - prefer (variant, strategy) consistent across both Top-2 picks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["C1f", "C1j"])
    ap.add_argument("--max_drop_pp", type=float, default=1.0,
                    help="Winner must have acc >= baseline - max_drop_pp")
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT / "models" / "early_exit_comparison.csv"))
    ap.add_argument("--out_winners", default=str(PROJECT_ROOT / "models" / "early_exit_winners.json"))
    args = ap.parse_args()

    rows = []
    baselines = {}
    for V in args.variants:
        path = PROJECT_ROOT / "models" / f"early_exit_{V}_ptq_v2.json"
        if not path.exists():
            print(f"WARN: {path} not found, skipping")
            continue
        data = json.loads(path.read_text())
        baselines[V] = data["baseline_acc"]
        for r in data["results"]:
            if r["strategy"] == "BASELINE_NOEXIT":
                continue
            rows.append({
                "variant": V,
                "strategy": r["strategy"],
                "threshold": r["threshold"],
                "min_exit_frame": r.get("min_exit_frame", 0),
                "accuracy": r["accuracy"],
                "mean_obs_ratio": r["mean_obs_ratio"],
                "latency_ms": r.get("latency_ms", 0.0),
                "baseline_acc": data["baseline_acc"],
                "acc_drop_pp": (data["baseline_acc"] - r["accuracy"]) * 100,
            })

    # Write all operating points
    header = ["variant", "strategy", "threshold", "min_exit_frame",
              "accuracy", "mean_obs_ratio", "latency_ms",
              "baseline_acc", "acc_drop_pp"]
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(
            f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]) for h in header
        ))
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(rows)} rows)")

    # Pick winner per variant: smallest obs_ratio s.t. acc_drop_pp <= max_drop_pp
    winners = {}
    print(f"\nWinner picking (max drop = {args.max_drop_pp} pp):")
    for V in args.variants:
        candidates = [r for r in rows
                      if r["variant"] == V and r["acc_drop_pp"] <= args.max_drop_pp]
        if not candidates:
            print(f"  {V}: NO winner — no operating point within {args.max_drop_pp} pp of baseline")
            winners[V] = None
            continue
        # Sort by mean_obs_ratio (latency proxy), then by acc (tiebreak)
        candidates.sort(key=lambda r: (r["mean_obs_ratio"], -r["accuracy"]))
        w = candidates[0]
        winners[V] = w
        print(f"  {V} winner: {w['strategy']} thresh={w['threshold']:.3f} "
              f"mf={w['min_exit_frame']}  acc={w['accuracy']:.4f} (drop {w['acc_drop_pp']:.2f}pp) "
              f"obs={w['mean_obs_ratio']:.3f}  lat={w['latency_ms']:.1f}ms")

    Path(args.out_winners).write_text(json.dumps({
        "max_drop_pp": args.max_drop_pp,
        "baselines": baselines,
        "winners": winners,
    }, indent=2))
    print(f"wrote {args.out_winners}")


if __name__ == "__main__":
    main()
