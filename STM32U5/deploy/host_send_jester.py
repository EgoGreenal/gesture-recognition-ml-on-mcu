"""host_send_jester.py -- Day 13 host-side companion to deploy/main_template.c.

Feeds Jester val clips frame-by-frame over UART to STM32U5 board running
C1j INT8 streaming + S1 mf=5 thresh=0.85 early-exit. Compares board predictions
against ground-truth labels + against host TFLite Interpreter, reports:
  - per-clip prediction agreement rate (board vs host)
  - accuracy (board vs label)
  - per-frame + per-clip latency (DWT cycle count -> ms at SYSCLK)
  - exit-frame distribution

Protocol with main_template.c:
  PC -> Board:
    b'S'                    one sync byte
    int8 frame (4096 B)     x8 (T=8 frames)
  Board -> PC (textual UTF-8 lines, '\n'-terminated):
    "F <t> <cycles>\n"      per frame (debug)
    "R <pred> <exit> <total_cycles>\n"   end of clip

Usage:
  python deploy/host_send_jester.py --port /dev/ttyUSB0 --baud 921600 --n_samples 100

Run on a machine that has the board's UART1 plugged in via USB-UART bridge.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import serial
from PIL import Image

PROJECT_ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from jester_common import H5_PATH, parse_split, sample_segment_indices

T_FRAMES = 8
FRAME_SIZE = 64 * 64 * 1
SYSCLK_HZ = 160_000_000


def quantize_frame_int8(frame_fp: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """frame_fp: (H, W) float in [-1, 1]. Returns (H, W) int8 quantized."""
    q = np.round(frame_fp / scale + zp).clip(-128, 127).astype(np.int8)
    return q


def load_val_clips_int8(n_samples: int, img_size: int = 64,
                         in_scale: float = 0.007843137718737125,
                         in_zp: int = -1):
    """Yield (frames_int8 [T,H,W], label, vid) tuples for first n val clips."""
    items = parse_split("validation")[:n_samples]
    with h5py.File(str(H5_PATH), "r") as h5:
        for vid, label in items:
            ds = h5["clips"][str(vid)]
            n_frames = ds.shape[0]
            offsets = sample_segment_indices(n_frames, T_FRAMES, training=False)
            all_bytes = ds[:]
            jpegs = [bytes(all_bytes[int(o)]) for o in offsets.tolist()]
            first = Image.open(io.BytesIO(jpegs[0])).convert("L")
            w, h = first.size
            side = min(w, h); top = (h - side) // 2; left = (w - side) // 2
            frames_int8 = np.empty((T_FRAMES, img_size, img_size), dtype=np.int8)
            for k, jpg in enumerate(jpegs):
                img = Image.open(io.BytesIO(jpg)).convert("L")
                img = img.crop((left, top, left + side, top + side)).resize(
                    (img_size, img_size), Image.BILINEAR
                )
                arr_fp = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                frames_int8[k] = quantize_frame_int8(arr_fp, in_scale, in_zp)
            yield frames_int8, label, vid


def host_tflite_eval(tflite_path: Path, frames_int8: np.ndarray) -> tuple[int, int]:
    """Run host TFLite Interpreter on same clip; return (pred, exit_frame_with_s1_mf5)."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    out_scale, out_zp = out["quantization"]

    # clip-form TFLite: input (1, T, H, W, 1), output (1, T, K)
    x = frames_int8.reshape(1, T_FRAMES, 64, 64, 1)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    q = interp.get_tensor(out["index"])  # (1, T, K) int8
    logits = (q.astype(np.float32) - out_zp) * out_scale  # (1, T, K)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # Apply S1 mf=5 thresh=0.85
    max_p = probs[0].max(axis=-1)  # (T,)
    fires = max_p > 0.85
    fires[:5] = False  # floor
    if fires.any():
        exit_frame = int(np.argmax(fires))
    else:
        exit_frame = T_FRAMES - 1
    pred = int(probs[0, exit_frame].argmax())
    return pred, exit_frame


def run_board_clip(ser: serial.Serial, frames_int8: np.ndarray) -> dict:
    """Send one clip to board, collect (pred, exit_frame, cycles) and per-frame cycles."""
    # Send sync byte + 8 frames
    ser.write(b"S")
    for t in range(T_FRAMES):
        ser.write(frames_int8[t].tobytes())  # 4096 bytes per frame
    ser.flush()

    # Receive per-frame debug + final R line
    frame_cycles = [None] * T_FRAMES
    pred = -1
    exit_frame = -1
    total_cycles = 0
    t0 = time.time()
    while True:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if not line:
            if time.time() - t0 > 30.0:
                raise TimeoutError("board did not respond in 30s")
            continue
        if line.startswith("F "):
            parts = line.split()
            t = int(parts[1])
            c = int(parts[2])
            if 0 <= t < T_FRAMES:
                frame_cycles[t] = c
        elif line.startswith("R "):
            parts = line.split()
            pred = int(parts[1])
            exit_frame = int(parts[2])
            total_cycles = int(parts[3])
            break
        else:
            # Stray debug -- forward to host stdout
            print(f"  [board debug] {line}", flush=True)
    return {
        "pred": pred,
        "exit_frame": exit_frame,
        "total_cycles": total_cycles,
        "frame_cycles": frame_cycles,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0", help="USART1 of board")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--n_samples", type=int, default=100, help="# Jester val clips to test")
    ap.add_argument("--tflite", default=str(PROJECT_ROOT / "models/student_C1J/deploy/C1j_ptq_clip_int8.tflite"),
                    help="Host TFLite for agreement check (clip-form)")
    ap.add_argument("--in_scale", type=float, default=0.007843137718737125)
    ap.add_argument("--in_zp", type=int, default=-1)
    ap.add_argument("--out_json", default=str(PROJECT_ROOT / "models/day13_board_eval.json"))
    args = ap.parse_args()

    print(f"[host] opening {args.port} @ {args.baud} baud", flush=True)
    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(2.0)
    ser.reset_input_buffer()

    n = 0
    n_correct_board = 0
    n_correct_host = 0
    n_agree = 0
    exit_hist = [0] * T_FRAMES
    total_cycles_acc = 0
    per_clip_records = []

    t_start = time.time()
    for frames, label, vid in load_val_clips_int8(args.n_samples,
                                                    in_scale=args.in_scale,
                                                    in_zp=args.in_zp):
        # Run on board
        board = run_board_clip(ser, frames)
        # Run on host (slow first time due to tf import; lazy)
        host_pred, host_exit = host_tflite_eval(Path(args.tflite), frames)

        is_correct_board = (board["pred"] == int(label))
        is_correct_host = (host_pred == int(label))
        agree = (board["pred"] == host_pred)
        n_correct_board += int(is_correct_board)
        n_correct_host += int(is_correct_host)
        n_agree += int(agree)
        exit_hist[board["exit_frame"]] += 1
        total_cycles_acc += board["total_cycles"]
        n += 1

        ms = board["total_cycles"] / SYSCLK_HZ * 1000.0
        marker = "✓" if is_correct_board else "✗"
        agree_marker = "=" if agree else "≠"
        print(f"  [{n:3d}/{args.n_samples}] vid={vid:6d} label={label:2d} "
              f"board={board['pred']:2d}{marker} host={host_pred:2d} {agree_marker} "
              f"exit_f={board['exit_frame']} cycles={board['total_cycles']:9d} "
              f"({ms:.1f} ms)",
              flush=True)

        per_clip_records.append({
            "vid": vid, "label": int(label),
            "board_pred": board["pred"],
            "host_pred": host_pred,
            "board_exit_frame": board["exit_frame"],
            "host_exit_frame": host_exit,
            "total_cycles": board["total_cycles"],
            "frame_cycles": board["frame_cycles"],
        })

    elapsed = time.time() - t_start
    print()
    print(f"== Summary ({n} clips, {elapsed:.1f}s wall) ==")
    print(f"  board accuracy   : {n_correct_board/n*100:.2f}% ({n_correct_board}/{n})")
    print(f"  host  accuracy   : {n_correct_host/n*100:.2f}% ({n_correct_host}/{n})")
    print(f"  board==host rate : {n_agree/n*100:.2f}% ({n_agree}/{n})")
    print(f"  mean total cyc   : {total_cycles_acc/n:.0f} ({total_cycles_acc/n/SYSCLK_HZ*1000:.2f} ms/clip @ {SYSCLK_HZ/1e6:.0f} MHz)")
    print(f"  exit_frame hist  : {exit_hist}")
    mean_obs = sum((i + 1) * c for i, c in enumerate(exit_hist)) / max(n, 1) / T_FRAMES
    print(f"  mean obs ratio   : {mean_obs:.3f}")

    import json
    Path(args.out_json).write_text(json.dumps({
        "n_samples": n,
        "board_accuracy": n_correct_board / n,
        "host_accuracy": n_correct_host / n,
        "agreement_rate": n_agree / n,
        "mean_total_cycles": total_cycles_acc / n,
        "sysclk_hz": SYSCLK_HZ,
        "mean_ms_per_clip": total_cycles_acc / n / SYSCLK_HZ * 1000.0,
        "exit_hist": exit_hist,
        "mean_obs_ratio": mean_obs,
        "per_clip": per_clip_records,
    }, indent=2))
    print(f"  saved to {args.out_json}")


if __name__ == "__main__":
    main()
