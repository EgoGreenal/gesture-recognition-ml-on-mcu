"""Day 11: run stedgeai analyze on an existing INT8 TFLite (no re-export).

Decoupled from `analyze_path_c.py` so it can be run on the PTQ AND QAT INT8
TFLite files produced by `export_int8.py`, without rebuilding/re-exporting
from the Keras model each time.

Parses stedgeai output -> dict (macc, ram_bytes, weights_bytes, rom_bytes).
Latency estimate per Plan formula:  ms_per_frame = macc / 300e6 * 1000 + 5.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")


def run_stedgeai(tflite_path: Path, out_dir: Path, name: str) -> dict:
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
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    elapsed = time.time() - t0
    log_text = result.stdout + "\n" + result.stderr
    (out_dir / "analyze_log.txt").write_text(log_text)

    def _find_int(pattern: str):
        m = re.search(pattern, log_text, re.IGNORECASE)
        return int(m.group(1).replace(",", "")) if m else None

    return {
        "returncode": result.returncode, "elapsed_s": round(elapsed, 2),
        "macc": _find_int(r"(?:total\s+)?(?:nb\s+of\s+)?macc[\s:]+([0-9,]+)"),
        "ram_bytes": _find_int(r"ram[^\d]+([0-9,]+)\s*(?:bytes|b)"),
        "rom_bytes": _find_int(r"rom[^\d]+([0-9,]+)\s*(?:bytes|b)"),
        "weights_bytes": _find_int(r"weights[^\d]+([0-9,]+)\s*(?:bytes|b)"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, help="INT8 TFLite file (streaming form)")
    ap.add_argument("--name", required=True, help="stedgeai network name (no spaces)")
    ap.add_argument("--out_dir", required=True, help="Workspace dir for stedgeai")
    ap.add_argument("--out_json", default=None, help="Append to this JSON list")
    ap.add_argument("--T", type=int, default=8, help="Frames per clip (for clip latency)")
    args = ap.parse_args()

    tflite_path = Path(args.tflite)
    if not tflite_path.exists():
        print(f"ERROR: TFLite not found: {tflite_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[analyze] {tflite_path.name}  -> {args.out_dir}")
    sai = run_stedgeai(tflite_path, Path(args.out_dir), args.name)
    macs = sai.get("macc") or 0
    lat_per_frame = macs / 300e6 * 1000 + 5
    lat_per_clip = lat_per_frame * args.T

    print(f"  rc={sai['returncode']}  elapsed={sai['elapsed_s']}s")
    print(f"  macc={sai.get('macc')}  ram={sai.get('ram_bytes')}  "
          f"rom={sai.get('rom_bytes')}  weights={sai.get('weights_bytes')}")
    print(f"  lat_per_frame={lat_per_frame:.2f} ms  lat_per_clip(T={args.T})="
          f"{lat_per_clip:.2f} ms")

    entry = {
        "tflite": str(tflite_path),
        "name": args.name,
        **sai,
        "lat_per_frame_ms": round(lat_per_frame, 2),
        "lat_per_clip_ms": round(lat_per_clip, 2),
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = (json.loads(out_path.read_text()) if out_path.exists() else [])
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(entry)
        out_path.write_text(json.dumps(existing, indent=2))
        print(f"  appended to {out_path}")


if __name__ == "__main__":
    main()
