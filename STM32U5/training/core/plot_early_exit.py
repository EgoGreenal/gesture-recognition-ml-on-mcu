"""Day 12: plot accuracy vs observation-ratio curves (TSM Fig 6 style)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(paths):
    out = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        out.append(data)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="early_exit_*.json files from early_exit_eval.py")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--title", default="Accuracy vs Observation Ratio (Jester val, INT8 PTQ)")
    args = ap.parse_args()

    datasets = load_results(args.inputs)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color cycle: variant determines color, strategy determines marker
    variant_colors = {}
    next_color_idx = 0
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    strategy_markers = {"S1": "o", "S3": "s"}

    for ds in datasets:
        variant = ds["variant"]
        tag = ds["tag"]
        baseline_acc = ds["baseline_acc"]
        if variant not in variant_colors:
            variant_colors[variant] = palette[next_color_idx % len(palette)]
            next_color_idx += 1
        col = variant_colors[variant]

        # Group by strategy
        for strat in ("S1", "S3"):
            pts = [(r["mean_obs_ratio"], r["accuracy"]) for r in ds["results"]
                   if r["strategy"] == strat]
            if not pts:
                continue
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker=strategy_markers[strat], color=col,
                    linestyle="--" if strat == "S3" else "-",
                    label=f"{variant}/{tag} {strat}", linewidth=1.5, markersize=6)
        # Baseline (no-exit) as star at (1.0, baseline_acc)
        ax.scatter([1.0], [baseline_acc], marker="*", color=col, s=180,
                   edgecolor="black", linewidth=0.8,
                   label=f"{variant}/{tag} baseline (no exit)", zorder=10)

    ax.set_xlabel("Mean observation ratio  (exit_frame + 1) / T")
    ax.set_ylabel("Accuracy (top-1)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
