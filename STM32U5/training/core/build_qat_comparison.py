"""Day 11: aggregate FP32 / PTQ / QAT INT8 numbers into models/qat_comparison.csv.

Reads from:
    * FP32 baseline:  models/student_<V>/<job_id>/metrics.json  (Day 10)
    * PTQ INT8:       /tmp/eval_int8_results.json  (eval_int8.py output, tag=ptq)
    * QAT INT8:       /tmp/eval_int8_results.json  (eval_int8.py output, tag=qat)
    * stedgeai:       /tmp/stedgeai_results.json   (analyze_int8.py output)

Output columns:
    variant, params, fp32_obs_1.00,
    ptq_int8_obs_1.00, ptq_drop, ptq_tflite_kb,
    qat_int8_obs_1.00, qat_drop, qat_tflite_kb,
    macs_per_call, ram_kb, weights_kb, lat_per_frame_ms,
    flash_ok, sram_ok, lat_ok,
    decision_tier   (default / risk / extend / fallback)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")

# Day 10 FP32 ckpt -> training job id (for finding metrics.json)
FP32_RUNS = {
    "C1f": "41478885_2",
    "C1j": "41478885_6",
}


def tier_for_int8_acc(acc: float) -> str:
    if acc >= 0.85:
        return "default"
    if acc >= 0.75:
        return "risk"
    if acc >= 0.60:
        return "extend"
    return "fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", default=None,
                    help="Single JSON list with all (variant, tag) entries. "
                         "If unset, falls back to per-variant: "
                         "models/eval_int8_<V>_<tag>.json")
    ap.add_argument("--analyze_json", default=None,
                    help="Single JSON list with all analyze entries. If unset, "
                         "falls back to per-variant: models/stedgeai_<V>_ptq.json")
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT / "models" / "qat_comparison.csv"))
    ap.add_argument("--variants", nargs="+", default=list(FP32_RUNS.keys()))
    ap.add_argument("--sram_budget_kb", type=int, default=600)
    ap.add_argument("--flash_budget_kb", type=int, default=2048)
    ap.add_argument("--lat_budget_ms_per_frame", type=float, default=150.0)
    args = ap.parse_args()

    # Collect eval data (per-variant JSONs if no aggregate provided)
    eval_data = []
    if args.eval_json:
        if Path(args.eval_json).exists():
            eval_data = json.loads(Path(args.eval_json).read_text())
    else:
        for V in args.variants:
            for tag in ("ptq", "qat"):
                f = PROJECT_ROOT / "models" / f"eval_int8_{V}_{tag}.json"
                if f.exists():
                    data = json.loads(f.read_text())
                    eval_data.extend(data if isinstance(data, list) else [data])

    # Collect analyze data
    analyze_data = []
    if args.analyze_json:
        if Path(args.analyze_json).exists():
            analyze_data = json.loads(Path(args.analyze_json).read_text())
    else:
        for V in args.variants:
            f = PROJECT_ROOT / "models" / f"stedgeai_{V}_ptq.json"
            if f.exists():
                data = json.loads(f.read_text())
                # Tag entries with variant for matching
                for e in (data if isinstance(data, list) else [data]):
                    e["_variant"] = V
                    analyze_data.append(e)

    rows = []
    for V in args.variants:
        # FP32 baseline
        run_id = FP32_RUNS.get(V)
        metrics_path = PROJECT_ROOT / "models" / f"student_{V.upper()}" / run_id / "metrics.json"
        m = json.loads(metrics_path.read_text())
        fp32_acc = m["best_val_obs_1.00"]
        n_params = m["config"]["n_params"]

        # PTQ / QAT INT8
        ptq = next((e for e in eval_data
                    if e["variant"].lower() == V.lower() and e.get("tag") == "ptq"), None)
        qat = next((e for e in eval_data
                    if e["variant"].lower() == V.lower() and e.get("tag") == "qat"), None)

        # stedgeai per-variant
        sai = next((e for e in analyze_data
                    if e.get("_variant", "").lower() == V.lower()
                    or V.lower() in str(e.get("name", "")).lower()
                    or V.lower() in str(e.get("tflite", "")).lower()), None)

        ptq_acc = ptq["obs_1.00"] if ptq else None
        qat_acc = qat["obs_1.00"] if qat else None
        ptq_drop = (fp32_acc - ptq_acc) if ptq_acc is not None else None
        qat_drop = (fp32_acc - qat_acc) if qat_acc is not None else None
        # Decide tier from QAT if present, else PTQ
        decision_acc = qat_acc if qat_acc is not None else ptq_acc
        tier = tier_for_int8_acc(decision_acc) if decision_acc is not None else "unknown"

        # Hardware constraints (from stedgeai). stedgeai analyzed the FP32-fallback
        # streaming TFLite (INT8 PTQ multi-input fails -- see Day 10/11 lessons);
        # so reported weights_bytes is FP32 weights. For INT8 deployment, the
        # actual Flash usage is weights_bytes / 4 (FP32 -> INT8 weight scaling).
        macs = (sai or {}).get("macc") or 0
        ram_b = (sai or {}).get("ram_bytes") or 0
        weights_b_fp32 = (sai or {}).get("weights_bytes") or 0
        weights_b_int8 = weights_b_fp32 // 4 if weights_b_fp32 else 0
        lat_fr = (sai or {}).get("lat_per_frame_ms") or 0.0
        flash_ok = (weights_b_int8 / 1024) <= args.flash_budget_kb if weights_b_int8 else True
        sram_ok = (ram_b / 1024) <= args.sram_budget_kb if ram_b else True
        lat_ok = lat_fr <= args.lat_budget_ms_per_frame if lat_fr else True

        ptq_tflite_kb = 0
        if ptq and ptq.get("tflite_path"):
            p = Path(ptq["tflite_path"])
            if p.exists():
                ptq_tflite_kb = round(p.stat().st_size / 1024, 1)

        row = {
            "variant": V,
            "params": n_params,
            "fp32_obs_1.00": round(fp32_acc, 4),
            "ptq_int8_obs_1.00": round(ptq_acc, 4) if ptq_acc is not None else "",
            "ptq_drop_pp": round(ptq_drop * 100, 2) if ptq_drop is not None else "",
            "ptq_clip_tflite_kb": ptq_tflite_kb,
            "qat_int8_obs_1.00": round(qat_acc, 4) if qat_acc is not None else "skipped",
            "qat_drop_pp": round(qat_drop * 100, 2) if qat_drop is not None else "skipped",
            "macs_per_call": macs,
            "ram_kb": round(ram_b / 1024, 1) if ram_b else "",
            "weights_fp32_kb": round(weights_b_fp32 / 1024, 1) if weights_b_fp32 else "",
            "weights_int8_kb_est": round(weights_b_int8 / 1024, 1) if weights_b_int8 else "",
            "lat_per_frame_ms": lat_fr if lat_fr else "",
            "flash_ok_int8": "Y" if flash_ok else "N",
            "sram_ok": "Y" if sram_ok else "N",
            "lat_ok": "Y" if lat_ok else "N",
            "decision_tier": tier,
        }
        rows.append(row)

    header = list(rows[0].keys())
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[h]).replace(",", ";") for h in header))
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\nsummary:")
    for r in rows:
        print(f"  {r['variant']}: FP32={r['fp32_obs_1.00']:.4f}  "
              f"PTQ={r['ptq_int8_obs_1.00']}  QAT={r['qat_int8_obs_1.00']}  "
              f"tier={r['decision_tier']}")


if __name__ == "__main__":
    main()
