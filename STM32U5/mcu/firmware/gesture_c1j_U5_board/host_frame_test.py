#!/usr/bin/env python3
"""host_frame_test.py - drive the F (real frame) command on C1j-ai-v2 firmware.

Usage examples (Windows PowerShell):
    pip install pyserial numpy
    python host_frame_test.py
    python host_frame_test.py --n-frames 8 --pattern tframe
    python host_frame_test.py --pattern zeros --no-reset

Protocol recap (per firmware main.c):
    'I' -> board zeros frame + all 10 cache_in tensors, replies "AI_RESET ok"
    'F' -> board then BLOCKS reading 4096 INT8 bytes, runs ai_run, replies:
             "INF cycles=<u32> us=<u32> argmax=<0..26> v=<int8>\r\n"
             "LOGITS <27 space-separated int8 values>\r\n"

The board's int8 quantization for the frame tensor is scale=0.007843138,
zero_point=-1 (per stedgeai report).  Float [-1, 1] maps to roughly [-128, 126].
"""

import argparse
import sys
import time

import numpy as np
import serial


PORT_DEFAULT = "COM4"
BAUD_DEFAULT = 115200
FRAME_BYTES  = 4096            # 64 * 64 * INT8, ai_input[0]
FRAME_H = FRAME_W = 64
N_LOGITS     = 27

# Quantization params for serving_default_frame_in0
Q_SCALE = 0.007843138
Q_ZP    = -1


def read_until(ser, marker, timeout):
    """Read whole lines until one starts with `marker` (bytes); return all lines.

    Each line is returned without trailing \r\n.  If timeout fires before the
    marker is seen, returns whatever was collected so far.
    """
    out_lines = []
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
            out_lines.append(line)
            if line.startswith(marker):
                return out_lines
    return out_lines


def make_frame(kind: str, t: int) -> bytes:
    """Return FRAME_BYTES bytes of INT8 payload, NHWC row-major (H*W*1)."""
    if kind == "zeros":
        arr = np.zeros(FRAME_BYTES, dtype=np.int8)
    elif kind == "ones":
        # Quantized "+1.0": q = round(1/scale) + zp = 127 + (-1) = 126
        arr = np.full(FRAME_BYTES, 126, dtype=np.int8)
    elif kind == "minus_ones":
        # Quantized "-1.0": -128
        arr = np.full(FRAME_BYTES, -128, dtype=np.int8)
    elif kind == "ramp":
        # Repeating 0..255 -> int8 by subtracting 128 (just to vary the input).
        arr = ((np.arange(FRAME_BYTES, dtype=np.int32) & 0xFF) - 128).astype(np.int8)
    elif kind == "tframe":
        # Per-frame seeded ramp so cache_in state visibly drifts across frames.
        seed = (t * 37) & 0xFF
        arr = (((np.arange(FRAME_BYTES, dtype=np.int32) + seed) & 0xFF) - 128).astype(np.int8)
    elif kind == "rand":
        rng = np.random.default_rng(seed=12345 + t)
        arr = rng.integers(-128, 128, size=FRAME_BYTES, dtype=np.int8)
    elif kind == "checker":
        # 8x8 black/white blocks at 64x64 resolution, mapped to -128/+126.
        img = np.zeros((FRAME_H, FRAME_W), dtype=np.int8)
        for y in range(FRAME_H):
            for x in range(FRAME_W):
                img[y, x] = 126 if (((x // 8) + (y // 8) + t) & 1) else -128
        arr = img.reshape(-1)
    else:
        raise ValueError(f"unknown pattern: {kind}")
    assert arr.size == FRAME_BYTES
    return arr.tobytes()


def parse_logits_line(line: bytes):
    """Parse 'LOGITS v0 v1 ... v26' into a list[int]; return None on mismatch."""
    if not line.startswith(b"LOGITS"):
        return None
    parts = line.split()
    if len(parts) != 1 + N_LOGITS:
        return None
    try:
        return [int(x) for x in parts[1:]]
    except ValueError:
        return None


def reset_board(ser, timeout=1.5):
    ser.reset_input_buffer()
    ser.write(b"I")
    ser.flush()
    lines = read_until(ser, marker=b"AI_RESET", timeout=timeout)
    for ln in lines:
        print(f"<  {ln.decode(errors='replace')}")
    if not any(ln.startswith(b"AI_RESET") for ln in lines):
        print("!! board did not acknowledge AI_RESET within timeout", file=sys.stderr)


def send_frame(ser, payload: bytes, gap: float, timeout: float):
    assert len(payload) == FRAME_BYTES
    ser.write(b"F")
    ser.flush()
    # Tiny gap so the board enters HAL_UART_Receive(4096) before any payload
    # byte hits RDR. With FIFO disabled a >87 us spacing is enough; default 10 ms
    # is very conservative and only costs throughput at very high frame rates.
    if gap > 0:
        time.sleep(gap)
    # Write the payload; pyserial chunks internally. flush() blocks until the
    # OS-side buffer is drained, not until the board has consumed all bytes,
    # but that's fine because read_until() then waits for the response.
    ser.write(payload)
    ser.flush()
    return read_until(ser, marker=b"LOGITS", timeout=timeout)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=PORT_DEFAULT)
    ap.add_argument("--baud", type=int, default=BAUD_DEFAULT)
    ap.add_argument("--n-frames", type=int, default=8,
                    help="streaming clip length (default: 8)")
    ap.add_argument("--pattern", default="tframe",
                    choices=["zeros", "ones", "minus_ones", "ramp",
                             "tframe", "rand", "checker"],
                    help="frame content (default: tframe)")
    ap.add_argument("--gap", type=float, default=0.010,
                    help="seconds between 'F' byte and 4096-byte payload "
                         "(default: 0.010)")
    ap.add_argument("--timeout", type=float, default=3.0,
                    help="seconds to wait for INF/LOGITS reply (default: 3.0)")
    ap.add_argument("--no-reset", action="store_true",
                    help="skip the leading 'I' reset (keep current cache state)")
    args = ap.parse_args()

    print(f"opening {args.port} @ {args.baud}")
    with serial.Serial(args.port, args.baud, timeout=0.05,
                       write_timeout=5.0) as ser:
        time.sleep(0.2)
        ser.reset_input_buffer()

        if not args.no_reset:
            print("-- reset --")
            reset_board(ser)

        argmaxes = []
        cycles_list = []
        for t in range(args.n_frames):
            payload = make_frame(args.pattern, t)
            print(f"-- frame {t}  (pattern={args.pattern}, {len(payload)} B) --")
            lines = send_frame(ser, payload, args.gap, args.timeout)

            saw_logits = False
            for ln in lines:
                txt = ln.decode(errors="replace")
                print(f"<  {txt}")
                if txt.startswith("INF "):
                    # parse cycles=NNN
                    for tok in txt.split():
                        if tok.startswith("cycles="):
                            cycles_list.append(int(tok.split("=", 1)[1]))
                        elif tok.startswith("argmax="):
                            argmaxes.append(int(tok.split("=", 1)[1]))
                if ln.startswith(b"LOGITS"):
                    saw_logits = True
                    vals = parse_logits_line(ln)
                    if vals is None:
                        print("!! malformed LOGITS line", file=sys.stderr)
            if not saw_logits:
                print(f"!! frame {t}: did not get LOGITS within {args.timeout}s",
                      file=sys.stderr)

        if cycles_list:
            print()
            print("== summary ==")
            print(f"frames    : {len(cycles_list)}")
            print(f"argmaxes  : {argmaxes}")
            print(f"cycles    : min={min(cycles_list)} max={max(cycles_list)} "
                  f"mean={sum(cycles_list)//len(cycles_list)}")


if __name__ == "__main__":
    main()
