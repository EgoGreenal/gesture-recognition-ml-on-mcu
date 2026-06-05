#!/usr/bin/env python3
"""host_benchmark.py - end-to-end C1j benchmark over the F/I protocol.

Sends real Jester validation clips frame-by-frame to the board, parses
INF/LOGITS replies, applies a host-side S1 early-exit policy, and reports
board-side accuracy + latency. Only the board cycle count goes into the
latency metric -- serial wire time is excluded by design (the user asked
for this explicitly).

Expected input data (--data, .npz) :
    clips_int8 : (N, T=8, 64, 64) int8   pre-quantized frames
                                         (scale=0.007843138, zp=-1)
    labels     : (N,)              int32 ground-truth class index in [0, 27)
    vids       : (N,)              int32 (OPTIONAL) Jester video IDs
    class_names: (27,)             str   (OPTIONAL) for per-class report

If you don't have this file yet, run scripts/dump_val_clips_int8.py
on Leonardo (where jester_full.h5 lives) and scp the resulting npz down.

Smoke-test mode without any data:
    python host_benchmark.py --synthetic 1
    python host_benchmark.py --synthetic 10

Real run ramp:
    python host_benchmark.py --data jester_val_int8.npz --max-samples 1
    python host_benchmark.py --data jester_val_int8.npz --max-samples 10
    python host_benchmark.py --data jester_val_int8.npz --max-samples 100 \\
                             --save-json bench_100.json
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import serial


# ---- tensor / model constants (locked to current firmware C1j-ai-v2) -------
FRAME_H = FRAME_W       = 64
FRAME_BYTES             = FRAME_H * FRAME_W      # 4096, INT8, NHWC row-major
T_FRAMES                = 8
N_LOGITS                = 27

# Output tensor quantization (gemm_93 line in
# X-CUBE-AI/App/student_c1j_ptq_int8_generate_report.txt).
OUT_SCALE_DEFAULT       = 0.187850028
OUT_ZP_DEFAULT          = 66

# Board clock (current FW: 160 MHz). Override with --sysclk-hz if you change it.
SYSCLK_HZ_DEFAULT       = 160_000_000

# Early-exit defaults (Day 12 S1 policy in Project_Plan_ch.md).
MIN_EXIT_FRAME_DEFAULT  = 5
THRESHOLD_DEFAULT       = 0.85


# ---------------------------------------------------------------------------
# UART I/O helpers
# ---------------------------------------------------------------------------
def read_until(ser: serial.Serial, marker: bytes, timeout: float) -> list[bytes]:
    """Read whole \\r\\n-terminated lines until one starts with `marker`.

    Returns the list of lines collected (without trailing \\r\\n). If the
    timeout fires before the marker is seen, returns whatever was collected.
    """
    out: list[bytes] = []
    buf = b""
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        chunk = ser.read(max(1, ser.in_waiting))
        if not chunk:
            continue
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.rstrip(b"\r")
            out.append(line)
            if line.startswith(marker):
                return out
    return out


def board_reset(ser: serial.Serial, timeout: float = 1.5) -> None:
    ser.reset_input_buffer()
    ser.write(b"I")
    ser.flush()
    lines = read_until(ser, b"AI_RESET", timeout)
    if not any(ln.startswith(b"AI_RESET") for ln in lines):
        raise TimeoutError(f"board did not acknowledge AI_RESET; got: {lines}")


def board_run_frame(ser: serial.Serial,
                    frame_int8: bytes,
                    gap: float,
                    timeout: float) -> dict:
    """Send F + 4096B, parse the reply, return {cycles, argmax_int8, logits}."""
    if len(frame_int8) != FRAME_BYTES:
        raise ValueError(f"frame must be exactly {FRAME_BYTES} bytes, got {len(frame_int8)}")
    ser.write(b"F")
    ser.flush()
    # >87 us spacing guarantees the board enters HAL_UART_Receive(4096) before
    # any payload byte hits RDR (USART1 FIFO is disabled).
    if gap > 0:
        time.sleep(gap)
    ser.write(frame_int8)
    ser.flush()

    lines = read_until(ser, b"LOGITS", timeout)
    cycles: int | None = None
    argmax_int8: int | None = None
    logits: np.ndarray | None = None
    err_line: bytes | None = None

    for ln in lines:
        if ln.startswith(b"INF "):
            for tok in ln.decode("ascii", "replace").split():
                if tok.startswith("cycles="):
                    cycles = int(tok.split("=", 1)[1])
                elif tok.startswith("argmax="):
                    argmax_int8 = int(tok.split("=", 1)[1])
        elif ln.startswith(b"LOGITS"):
            parts = ln.split()
            if len(parts) == 1 + N_LOGITS:
                logits = np.fromiter((int(x) for x in parts[1:]),
                                     dtype=np.int32, count=N_LOGITS).astype(np.int8)
        elif ln.startswith(b"FRAME_RX_FAIL") or ln.startswith(b"AI"):
            err_line = ln

    if cycles is None or logits is None:
        raise IOError(f"malformed reply (err={err_line!r}, lines={lines})")
    return {"cycles": cycles, "argmax_int8": argmax_int8, "logits": logits}


# ---------------------------------------------------------------------------
# Host-side numerics
# ---------------------------------------------------------------------------
def dequant_logits(logits_int8: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return (logits_int8.astype(np.float32) - zp) * scale


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(args) -> tuple[np.ndarray, np.ndarray, list[str] | None,
                              np.ndarray | None]:
    """Returns (clips_int8 [N,T,H,W], labels [N], class_names | None, vids | None)."""
    if args.synthetic > 0:
        rng = np.random.default_rng(0)
        N = args.synthetic
        clips = rng.integers(-128, 128,
                             size=(N, T_FRAMES, FRAME_H, FRAME_W),
                             dtype=np.int8)
        labels = rng.integers(0, N_LOGITS, size=(N,), dtype=np.int32)
        return clips, labels, None, None

    p = Path(args.data)
    if not p.exists():
        sys.exit(
            f"!! data file not found: {p}\n"
            f"   either pass --synthetic N for a smoke run, or provide a .npz with:\n"
            f"     clips_int8 (N,{T_FRAMES},{FRAME_H},{FRAME_W}) int8\n"
            f"     labels     (N,) int\n"
            f"   See scripts/dump_val_clips_int8.py for a Leonardo-side dumper."
        )
    d = np.load(p, allow_pickle=False)
    if "clips_int8" not in d.files:
        sys.exit(f"!! {p} missing 'clips_int8'; got keys {list(d.files)}")
    if "labels" not in d.files:
        sys.exit(f"!! {p} missing 'labels'; got keys {list(d.files)}")
    clips = d["clips_int8"]
    labels = d["labels"].astype(np.int32)
    if clips.dtype != np.int8:
        sys.exit(f"!! clips_int8 dtype must be int8, got {clips.dtype}")
    if clips.shape[1:] != (T_FRAMES, FRAME_H, FRAME_W):
        sys.exit(f"!! clips_int8 shape must be (N,{T_FRAMES},{FRAME_H},{FRAME_W}); "
                 f"got {clips.shape}")
    if labels.shape != (clips.shape[0],):
        sys.exit(f"!! labels shape {labels.shape} mismatches clips N={clips.shape[0]}")
    class_names = ([str(x) for x in d["class_names"]]
                   if "class_names" in d.files else None)
    vids = d["vids"].astype(np.int64) if "vids" in d.files else None
    return clips, labels, class_names, vids


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------
def run_one_clip(ser, clip: np.ndarray, *,
                 out_scale: float, out_zp: int,
                 min_exit_frame: int, threshold: float,
                 gap: float, frame_timeout: float,
                 verbose: bool = False) -> dict:
    """Drive one clip through the board. Returns a record dict."""
    board_reset(ser)

    per_frame_cycles: list[int] = []
    per_frame_pmax: list[float] = []
    per_frame_argmax: list[int] = []
    exit_frame = T_FRAMES - 1
    pred = -1
    cum_cycles_at_exit = 0

    for t in range(T_FRAMES):
        r = board_run_frame(ser, clip[t].tobytes(), gap, frame_timeout)
        per_frame_cycles.append(r["cycles"])

        logits_fp = dequant_logits(r["logits"], out_scale, out_zp)
        p = softmax(logits_fp)
        p_max = float(p.max())
        cls = int(p.argmax())
        per_frame_pmax.append(p_max)
        per_frame_argmax.append(cls)

        if verbose:
            print(f"      f{t}: cyc={r['cycles']:>9d} "
                  f"argmax_int8={r['argmax_int8']:>2d} "
                  f"argmax_sm={cls:>2d} p_max={p_max:.3f}")

        if t >= min_exit_frame and p_max > threshold:
            pred = cls
            exit_frame = t
            cum_cycles_at_exit = sum(per_frame_cycles)
            break
    else:
        # No exit fired -> use last frame's prediction
        pred = per_frame_argmax[-1]
        exit_frame = T_FRAMES - 1
        cum_cycles_at_exit = sum(per_frame_cycles)

    return {
        "pred": pred,
        "exit_frame": exit_frame,
        "cum_cycles": cum_cycles_at_exit,
        "per_frame_cycles": per_frame_cycles,
        "per_frame_pmax": per_frame_pmax,
        "per_frame_argmax": per_frame_argmax,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--data", default="jester_val_int8.npz",
                    help="path to .npz with clips_int8 + labels (default: %(default)s)")
    ap.add_argument("--max-samples", type=int, default=1,
                    help="number of clips to evaluate (default: 1)")
    ap.add_argument("--min-exit-frame", type=int, default=MIN_EXIT_FRAME_DEFAULT,
                    help="floor frame index for early exit (default: %(default)s)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                    help="softmax-max threshold for early exit (default: %(default)s)")
    ap.add_argument("--out-scale", type=float, default=OUT_SCALE_DEFAULT,
                    help="output INT8 dequant scale (default: %(default)s)")
    ap.add_argument("--out-zp", type=int, default=OUT_ZP_DEFAULT,
                    help="output INT8 dequant zero point (default: %(default)s)")
    ap.add_argument("--sysclk-hz", type=int, default=SYSCLK_HZ_DEFAULT,
                    help="board SYSCLK for cycles->ms conversion (default: %(default)s)")
    ap.add_argument("--gap", type=float, default=0.010,
                    help="seconds between 'F' byte and 4096-byte payload "
                         "(default: %(default)s)")
    ap.add_argument("--frame-timeout", type=float, default=5.0,
                    help="seconds to wait for INF/LOGITS reply per frame "
                         "(default: %(default)s)")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="ignore --data and use N synthetic random clips")
    ap.add_argument("--save-json", default=None,
                    help="if set, write per-clip records + summary to this path")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-frame INF/LOGITS lines")
    args = ap.parse_args()

    clips, labels, class_names, vids = load_data(args)
    n_clips = min(args.max_samples, len(clips))
    if n_clips <= 0:
        sys.exit("!! no clips to evaluate")

    print(f"data    : clips={tuple(clips.shape)} dtype={clips.dtype} "
          f"labels={tuple(labels.shape)}")
    print(f"policy  : min_exit_frame={args.min_exit_frame} "
          f"threshold={args.threshold}")
    print(f"output  : out_scale={args.out_scale} out_zp={args.out_zp}")
    print(f"clock   : sysclk={args.sysclk_hz/1e6:.1f} MHz")
    print(f"link    : {args.port} @ {args.baud}")
    print()

    records = []
    n_correct = 0
    per_class: dict[int, list[int]] = defaultdict(lambda: [0, 0])  # cls -> [correct, total]
    exit_hist = [0] * T_FRAMES
    total_cycles_sum = 0

    with serial.Serial(args.port, args.baud, timeout=0.05,
                       write_timeout=10.0) as ser:
        time.sleep(0.2)
        ser.reset_input_buffer()

        t_wall0 = time.monotonic()
        for i in range(n_clips):
            label = int(labels[i])
            vid = int(vids[i]) if vids is not None else -1
            try:
                rec = run_one_clip(
                    ser, clips[i],
                    out_scale=args.out_scale, out_zp=args.out_zp,
                    min_exit_frame=args.min_exit_frame, threshold=args.threshold,
                    gap=args.gap, frame_timeout=args.frame_timeout,
                    verbose=args.verbose)
            except (TimeoutError, IOError) as e:
                print(f"  [{i+1:4d}/{n_clips}] vid={vid} label={label} "
                      f"!! board I/O error: {e}")
                continue

            ok = (rec["pred"] == label)
            n_correct += int(ok)
            exit_hist[rec["exit_frame"]] += 1
            total_cycles_sum += rec["cum_cycles"]
            per_class[label][0] += int(ok)
            per_class[label][1] += 1

            ms = rec["cum_cycles"] / args.sysclk_hz * 1000.0
            tag = "OK" if ok else "MS"
            vid_str = f"vid={vid:6d} " if vid >= 0 else ""
            print(f"  [{i+1:4d}/{n_clips}] {vid_str}label={label:2d} "
                  f"pred={rec['pred']:2d} {tag} exit_f={rec['exit_frame']} "
                  f"cycles={rec['cum_cycles']:9d} ({ms:6.2f} ms)")

            records.append({
                "idx": i, "vid": vid, "label": label,
                "pred": rec["pred"], "ok": ok,
                "exit_frame": rec["exit_frame"],
                "cum_cycles": rec["cum_cycles"],
                "per_frame_cycles": rec["per_frame_cycles"],
                "per_frame_pmax": rec["per_frame_pmax"],
                "per_frame_argmax": rec["per_frame_argmax"],
            })
        wall_s = time.monotonic() - t_wall0

    n_done = len(records)
    if n_done == 0:
        sys.exit("!! no clips completed; nothing to summarize")

    acc = n_correct / n_done
    mean_exit = sum(t * c for t, c in enumerate(exit_hist)) / n_done
    mean_cycles = total_cycles_sum / n_done
    mean_ms = mean_cycles / args.sysclk_hz * 1000.0

    print()
    print("== summary ==")
    print(f"clips completed   : {n_done}/{n_clips}  (wall {wall_s:.1f}s)")
    print(f"accuracy          : {acc*100:.2f}%  ({n_correct}/{n_done})")
    print(f"mean exit frame   : {mean_exit:.2f}   (T={T_FRAMES})")
    print(f"mean obs ratio    : {(mean_exit+1)/T_FRAMES:.3f}")
    print(f"mean cycles/clip  : {mean_cycles:.0f}")
    print(f"mean latency/clip : {mean_ms:.2f} ms @ {args.sysclk_hz/1e6:.0f} MHz "
          f"(board cycles only -- serial wire time excluded)")
    print(f"exit-frame hist   : {exit_hist}")

    if per_class:
        print()
        print("per-class accuracy:")
        for cls in sorted(per_class):
            c, n = per_class[cls]
            name = (class_names[cls]
                    if class_names is not None and cls < len(class_names)
                    else f"class_{cls}")
            print(f"  {cls:2d}  {name:32s}  {c}/{n}  ({c/n*100:5.1f}%)")

    if args.save_json:
        import json
        payload = {
            "n_clips_requested": n_clips,
            "n_clips_completed": n_done,
            "accuracy": acc,
            "mean_exit_frame": mean_exit,
            "mean_obs_ratio": (mean_exit + 1) / T_FRAMES,
            "mean_cycles_per_clip": mean_cycles,
            "mean_ms_per_clip": mean_ms,
            "exit_hist": exit_hist,
            "wall_seconds": wall_s,
            "sysclk_hz": args.sysclk_hz,
            "min_exit_frame": args.min_exit_frame,
            "threshold": args.threshold,
            "out_scale": args.out_scale,
            "out_zp": args.out_zp,
            "per_class": {str(k): {"correct": v[0], "total": v[1]}
                          for k, v in per_class.items()},
            "records": records,
        }
        Path(args.save_json).write_text(json.dumps(payload, indent=2))
        print(f"\nsaved JSON to {args.save_json}")


if __name__ == "__main__":
    main()
