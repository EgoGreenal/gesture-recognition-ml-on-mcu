"""Realtime MAX78000 camera viewer.

This script reads the image/result packets emitted by the Exercise 8 camera firmware,
displays the RGB888/RGB565 frames as a live matplotlib view, and prints the
classification text. The 12 fps gesture firmware sends a result packet for every
inference and an image packet for every third inference.

Example:
    python realtime_camera_demo.py --port COM5

For a quick non-GUI smoke test:
    python realtime_camera_demo.py --port COM5 --no-display --frames 3 --save-dir ../camera_capture
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import serial


IMAGE_TOKEN = b"New image\n"
RESULT_TOKEN = b"New result\n"
TOKENS = (IMAGE_TOKEN, RESULT_TOKEN)


@dataclass
class FramePacket:
    width: int
    height: int
    pixel_format: str
    image: np.ndarray | None
    result: str
    has_image: bool


def read_exact(port: serial.Serial, length: int) -> bytes:
    """Read exactly length bytes or raise TimeoutError."""
    data = bytearray()
    deadline = time.monotonic() + max(port.timeout or 1.0, 1.0) * 4
    while len(data) < length:
        chunk = port.read(length - len(data))
        if chunk:
            data.extend(chunk)
            deadline = time.monotonic() + max(port.timeout or 1.0, 1.0) * 4
            continue
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out reading {length} bytes, got {len(data)}")
    return bytes(data)


def wait_for_packet_token(port: serial.Serial) -> bytes:
    """Synchronize to the next image or result packet token."""
    window = bytearray()
    max_token_len = max(len(token) for token in TOKENS)

    while True:
        byte = port.read(1)
        if not byte:
            raise TimeoutError("Timed out waiting for image/result token")

        window.append(byte[0])
        if len(window) > max_token_len:
            del window[0]

        for token in TOKENS:
            if window.endswith(token):
                return token


def read_packet(port: serial.Serial) -> FramePacket:
    """Read one image or result packet from the serial stream."""
    token = wait_for_packet_token(port)

    if token == RESULT_TOKEN:
        result_len = int.from_bytes(read_exact(port, 1), byteorder="big")
        result = read_exact(port, result_len).decode("ascii", errors="replace")
        return FramePacket(0, 0, "RESULT", None, result, False)

    header = read_exact(port, 5)
    width = int.from_bytes(header[0:2], byteorder="big")
    height = int.from_bytes(header[2:4], byteorder="big")
    format_len = header[4]
    pixel_format = read_exact(port, format_len).decode("ascii", errors="replace")

    image_len = int.from_bytes(read_exact(port, 4), byteorder="big")
    image_bytes = read_exact(port, image_len)

    result_len = int.from_bytes(read_exact(port, 1), byteorder="big")
    result = read_exact(port, result_len).decode("ascii", errors="replace")

    if pixel_format == "RGB888":
        expected = width * height * 3
        if image_len != expected:
            raise ValueError(f"Unexpected image length {image_len}; expected {expected}")
        image = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 3))
    elif pixel_format == "RGB565":
        expected = width * height * 2
        if image_len != expected:
            raise ValueError(f"Unexpected image length {image_len}; expected {expected}")
        raw = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height * width, 2))
        red = ((raw[:, 0] & 0xF8) >> 3).astype(np.uint16)
        green = (((raw[:, 0] & 0x07) << 3) | ((raw[:, 1] & 0xE0) >> 5)).astype(np.uint16)
        blue = (raw[:, 1] & 0x1F).astype(np.uint16)
        image = np.empty((height * width, 3), dtype=np.uint8)
        image[:, 0] = (red * 255 // 31).astype(np.uint8)
        image[:, 1] = (green * 255 // 63).astype(np.uint8)
        image[:, 2] = (blue * 255 // 31).astype(np.uint8)
        image = image.reshape((height, width, 3))
    elif pixel_format == "GRAY8":
        expected = width * height
        if image_len != expected:
            raise ValueError(f"Unexpected image length {image_len}; expected {expected}")
        gray = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width))
        image = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        raise ValueError(f"Unsupported pixel format {pixel_format!r}; expected RGB888, RGB565, or GRAY8")

    return FramePacket(width, height, pixel_format, image, result, True)


def summarize_result(result: str) -> str:
    """Return a compact one-line title from the firmware result text."""
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    label = lines[-1] if lines else "No result"
    matches = re.findall(r"Class\s+(\d+):\s+([0-9.]+)%", result)
    if matches:
        pred_class, confidence = max(matches, key=lambda item: float(item[1]))
        return f"{label} | class {pred_class} | {confidence}%"
    return label


def parse_result_metadata(result: str) -> dict[str, str]:
    """Extract key=value metadata from the firmware result text."""
    return dict(re.findall(r"(\w+)=([^\s]+)", result))


def parse_profile_metadata(metadata: dict[str, str]) -> dict[str, int]:
    """Parse firmware-side profiling fields from the metadata line."""
    profile_keys = ("capture", "get", "history", "load", "cnn", "softmax", "post", "total")
    raw_values = metadata.get("prof_us", "").split(",")
    if len(raw_values) != len(profile_keys):
        return {}

    try:
        parsed = {key: int(value) for key, value in zip(profile_keys, raw_values)}
        parsed["mhz"] = int(metadata.get("mhz", "0"))
    except ValueError:
        return {}

    return parsed


def format_profile_summary(metadata: dict[str, str]) -> str:
    """Return a compact timing/cycle summary for console output."""
    profile = parse_profile_metadata(metadata)
    if not profile:
        return ""

    mhz = max(profile.get("mhz", 0), 1)
    preproc_us = profile["get"] + profile["history"]
    def timing(name: str, elapsed_us: int) -> str:
        return f"{name}={elapsed_us}us/{elapsed_us * mhz}cy"

    return (
        " | profile "
        f"{timing('cap', profile['capture'])} "
        f"{timing('prep', preproc_us)} "
        f"{timing('load', profile['load'])} "
        f"{timing('cnn', profile['cnn'])} "
        f"{timing('softmax', profile['softmax'])} "
        f"{timing('post', profile['post'])} "
        f"{timing('total', profile['total'])}"
    )


def is_valid_result(result: str) -> bool:
    """Check that the text following a frame is really a classification payload."""
    return "Classification results:" in result and "Class" in result


def save_frame(packet: FramePacket, save_dir: Path, frame_index: int) -> None:
    """Save a frame as PPM plus the corresponding text result."""
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / f"frame_{frame_index:04d}.txt"
    txt_path.write_text(packet.result, encoding="utf-8")

    if packet.image is not None:
        ppm_path = save_dir / f"frame_{frame_index:04d}.ppm"
        header = f"P6\n{packet.width} {packet.height}\n255\n".encode("ascii")
        ppm_path.write_bytes(header + packet.image.tobytes())


def run(args: argparse.Namespace) -> None:
    packet_count = 0
    inference_count = 0
    image_count = 0
    start = time.monotonic()

    plot = None
    image_artist = None
    title = None
    if not args.no_display:
        import matplotlib.pyplot as plt

        plt.ion()
        plot = plt
        figure, axis = plt.subplots()
        figure.canvas.manager.set_window_title("MAX78000 Camera Demo")
        image_artist = axis.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
        title = axis.set_title("Waiting for frames...")
        axis.axis("off")
        figure.tight_layout()

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as port:
        port.reset_input_buffer()
        print(f"Opened {port.name} at {args.baud} baud")
        print("Waiting for camera packets... Press Ctrl+C to stop.")

        while args.frames == 0 or packet_count < args.frames:
            try:
                packet = read_packet(port)
            except (TimeoutError, ValueError) as exc:
                print(f"Resyncing after packet error: {exc}")
                continue

            if not is_valid_result(packet.result):
                preview = packet.result.replace("\n", "\\n")[:60]
                preview = preview.encode("ascii", errors="backslashreplace").decode("ascii")
                print(f"Skipping unsynchronized packet result: {preview!r}")
                continue

            packet_count += 1
            if packet.has_image:
                image_count += 1

            metadata = parse_result_metadata(packet.result)
            cnn_us = metadata.get("cnn_us")
            ran_cnn = cnn_us is not None and cnn_us.isdigit() and int(cnn_us) > 0
            if ran_cnn:
                inference_count += 1

            elapsed = max(time.monotonic() - start, 1e-6)
            inference_fps = inference_count / elapsed
            display_fps = image_count / elapsed
            summary = summarize_result(packet.result)
            prefix = metadata.get("prefix", "--")
            policy = metadata.get("policy", "--")
            cnn_text = f"{int(cnn_us) / 1000:.2f} ms" if ran_cnn and cnn_us else "skip"
            profile_text = format_profile_summary(metadata) if args.profile else ""
            if packet.has_image:
                size = f"{packet.width}x{packet.height}"
                packet_kind = "image"
            else:
                size = "--"
                packet_kind = "result"
            print(
                f"[P{packet_count:04d}/C{inference_count:04d}/D{image_count:04d}] "
                f"{packet_kind:6s} {size} p{prefix} {policy} {cnn_text} "
                f"{summary} (cnn {inference_fps:.2f} fps, display {display_fps:.2f} fps)"
                f"{profile_text}"
            )

            if args.save_dir:
                save_frame(packet, Path(args.save_dir), packet_count)

            if plot is not None and image_artist is not None and title is not None:
                if packet.image is not None:
                    image_artist.set_data(packet.image)
                title.set_text(
                    f"{summary} | p{prefix} {policy} | cnn {inference_fps:.2f} fps | display {display_fps:.2f} fps"
                )
                plot.pause(0.001)

    if plot is not None:
        plot.ioff()
        plot.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime MAX78000 camera viewer")
    parser.add_argument("--port", default="COM5", help="serial port connected to the MAX78000")
    parser.add_argument("--baud", type=int, default=2000000, help="serial baud rate")
    parser.add_argument("--timeout", type=float, default=2.0, help="serial read timeout in seconds")
    parser.add_argument("--frames", type=int, default=0, help="number of frames to read, 0 means forever")
    parser.add_argument("--save-dir", help="optional directory to save PPM frames and result text")
    parser.add_argument("--no-display", action="store_true", help="read packets without opening a GUI window")
    parser.add_argument("--profile", action="store_true", help="print firmware timing and approximate cycle counts")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\nStopped.")
