"""Tkinter realtime viewer for the MAX78000 Exercise 8 camera firmware.

This is a more reliable Windows-friendly viewer than matplotlib for long-running
interactive display. It reads result packets at the firmware inference rate and
updates the image only when an image packet arrives.

Example:
    python realtime_camera_tk_demo.py --port COM5
"""

from __future__ import annotations

import argparse
import queue
import re
import threading
import time
import tkinter as tk
from dataclasses import dataclass

from PIL import Image, ImageTk

from realtime_camera_demo import FramePacket, is_valid_result, parse_profile_metadata, read_packet

import serial


NEGATIVE_CLASSES = {"25", "26"}
DEFAULT_THRESHOLD_PERCENT = 55
LIGHT_OFF_COLOR = "#b8b8b8"
LIGHT_ON_COLOR = "#24b34b"
DIRECTION_COMMANDS = {
    "Swiping Left": "left",
    "Sliding Two Fingers Left": "left",
    "Swiping Right": "right",
    "Sliding Two Fingers Right": "right",
    "Swiping Up": "up",
    "Sliding Two Fingers Up": "up",
    "Swiping Down": "down",
    "Sliding Two Fingers Down": "down",
}


@dataclass
class ReaderStatus:
    message: str
    is_error: bool = False


def parse_result_fields(result: str) -> tuple[str, str, str, float | None, dict[str, str]]:
    """Extract stable UI fields from the firmware result text."""
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    label = lines[-1] if lines else "No result"

    class_match = re.search(r"Class\s+(\d+):\s+([0-9.]+)%", result)
    pred_class = class_match.group(1) if class_match else "--"
    confidence_value = float(class_match.group(2)) if class_match else None
    confidence = f"{confidence_value:.1f}%" if confidence_value is not None else "--"
    metadata = dict(re.findall(r"(\w+)=([^\s]+)", result))

    return label, pred_class, confidence, confidence_value, metadata


def serial_reader(
    port_name: str,
    baud: int,
    timeout: float,
    frame_queue: "queue.Queue[FramePacket | ReaderStatus]",
    stop_event: threading.Event,
) -> None:
    """Continuously read frames from the board in a background thread."""
    try:
        with serial.Serial(port_name, baud, timeout=timeout) as port:
            port.reset_input_buffer()
            frame_queue.put(ReaderStatus(f"Opened {port.name} at {baud} baud"))

            while not stop_event.is_set():
                try:
                    packet = read_packet(port)
                except (TimeoutError, ValueError) as exc:
                    frame_queue.put(ReaderStatus(f"Resyncing: {exc}"))
                    continue

                if not is_valid_result(packet.result):
                    continue

                frame_queue.put(packet)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        frame_queue.put(ReaderStatus(f"Serial reader stopped: {exc}", is_error=True))


class CameraApp:
    """Small Tkinter app for displaying live frames."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = tk.Tk()
        self.root.title("MAX78000 Realtime Camera Demo")
        self.root.configure(bg="#f4f4f4")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.frame_queue: "queue.Queue[FramePacket | ReaderStatus]" = queue.Queue()
        self.stop_event = threading.Event()
        self.reader = threading.Thread(
            target=serial_reader,
            args=(args.port, args.baud, args.timeout, self.frame_queue, self.stop_event),
            daemon=True,
        )

        self.start_time = time.monotonic()
        self.packet_count = 0
        self.inference_count = 0
        self.display_count = 0
        self.current_photo: ImageTk.PhotoImage | None = None
        self.video_size = 64 * args.scale
        self.panel_width = 420
        self.panel_height = max(self.video_size, 510 if args.profile else self.video_size)

        self.content = tk.Frame(self.root, bg="#f4f4f4")
        self.content.grid(row=0, column=0, padx=12, pady=12)

        self.image_frame = tk.Frame(
            self.content,
            width=self.video_size,
            height=self.video_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#222",
        )
        self.image_frame.grid(row=0, column=0, sticky="n")
        self.image_frame.grid_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="black")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        self.side_panel = tk.Frame(self.content, width=self.panel_width, height=self.panel_height, bg="#f4f4f4")
        self.side_panel.grid(row=0, column=1, padx=(12, 0), sticky="ns")
        self.side_panel.grid_propagate(False)

        self.prediction_var = tk.StringVar(value="Waiting")
        self.class_var = tk.StringVar(value="--")
        self.confidence_var = tk.StringVar(value="--")
        self.threshold_var = tk.DoubleVar(value=args.threshold)
        self.threshold_label_var = tk.StringVar(value=f"{args.threshold:.0f}%")
        self.last_positive_var = tk.StringVar(value="None yet")
        self.inference_fps_var = tk.StringVar(value="--")
        self.display_fps_var = tk.StringVar(value="--")
        self.cnn_time_var = tk.StringVar(value="--")
        self.capture_time_var = tk.StringVar(value="--")
        self.preprocess_time_var = tk.StringVar(value="--")
        self.load_time_var = tk.StringVar(value="--")
        self.softmax_time_var = tk.StringVar(value="--")
        self.total_time_var = tk.StringVar(value="--")
        self.cycle_count_var = tk.StringVar(value="--")
        self.packet_var = tk.StringVar(value="--")
        self.status_var = tk.StringVar(value=f"Port {args.port} | {args.baud} baud")
        self.light_boxes: dict[str, tk.Frame] = {}

        self.make_info_row(
            height=84,
            specs=(
                ("Prediction", self.prediction_var, 220, ("Segoe UI", 10, "bold")),
                ("Class", self.class_var, 82, ("Segoe UI", 16, "bold")),
                ("Confidence", self.confidence_var, 102, ("Segoe UI", 14, "bold")),
            ),
        )
        self.make_threshold_box()
        self.make_last_accepted_row()
        self.make_info_row(
            height=58,
            specs=(
                ("CNN Run FPS", self.inference_fps_var, 132, ("Segoe UI", 14, "bold")),
                ("Display FPS", self.display_fps_var, 132, ("Segoe UI", 14, "bold")),
                ("CNN Time", self.cnn_time_var, 132, ("Segoe UI", 12, "bold")),
            ),
        )
        if args.profile:
            self.make_info_row(
                height=54,
                specs=(
                    ("Capture", self.capture_time_var, 132, ("Segoe UI", 10, "bold")),
                    ("Preproc", self.preprocess_time_var, 132, ("Segoe UI", 10, "bold")),
                    ("Load", self.load_time_var, 132, ("Segoe UI", 10, "bold")),
                ),
            )
            self.make_info_row(
                height=54,
                specs=(
                    ("Softmax", self.softmax_time_var, 132, ("Segoe UI", 10, "bold")),
                    ("Total", self.total_time_var, 132, ("Segoe UI", 10, "bold")),
                    ("Cycles", self.cycle_count_var, 132, ("Segoe UI", 9, "bold")),
                ),
            )
        packet_label, self.status_label = self.make_info_row(
            height=64,
            specs=(
                ("Packet", self.packet_var, 204, ("Segoe UI", 9, "bold")),
                ("Status", self.status_var, 204, ("Segoe UI", 9)),
            ),
        )

    def make_info_box(
        self,
        title: str,
        variable: tk.StringVar,
        height: int = 46,
        font: tuple[str, int] | tuple[str, int, str] = ("Segoe UI", 12, "bold"),
        parent: tk.Widget | None = None,
        width: int | None = None,
        wraplength: int | None = None,
    ) -> tk.Label:
        parent = parent or self.side_panel
        width = width or self.panel_width
        box = tk.Frame(
            parent,
            width=width,
            height=height,
            bg="white",
            highlightbackground="#d0d0d0",
            highlightthickness=1,
        )
        if parent is self.side_panel:
            box.pack(fill="x", pady=(0, 6))
        else:
            box.pack(fill="both", expand=True)
        box.pack_propagate(False)

        tk.Label(
            box,
            text=title.upper(),
            bg="white",
            fg="#666",
            font=("Segoe UI", 7),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(4, 0))

        value_label = tk.Label(
            box,
            textvariable=variable,
            bg="white",
            fg="#111",
            font=font,
            anchor="w",
            justify="left",
            wraplength=wraplength or max(width - 22, 20),
        )
        value_label.pack(fill="both", expand=True, padx=10, pady=(0, 5))
        return value_label

    def make_info_row(
        self,
        height: int,
        specs: tuple[tuple[str, tk.StringVar, int, tuple[str, int] | tuple[str, int, str]], ...],
    ) -> tuple[tk.Label, ...]:
        row = tk.Frame(self.side_panel, width=self.panel_width, height=height, bg="#f4f4f4")
        row.pack(fill="x", pady=(0, 6))
        row.pack_propagate(False)

        labels = []
        for index, (title, variable, width, font) in enumerate(specs):
            cell = tk.Frame(row, width=width, height=height, bg="#f4f4f4")
            cell.pack(side="left", padx=(0, 8 if index < len(specs) - 1 else 0))
            cell.pack_propagate(False)
            labels.append(
                self.make_info_box(
                    title,
                    variable,
                    height=height,
                    font=font,
                    parent=cell,
                    width=width,
                    wraplength=max(width - 20, 20),
                )
            )

        return tuple(labels)

    def make_threshold_box(self) -> None:
        box = tk.Frame(
            self.side_panel,
            width=self.panel_width,
            height=60,
            bg="white",
            highlightbackground="#d0d0d0",
            highlightthickness=1,
        )
        box.pack(fill="x", pady=(0, 6))
        box.pack_propagate(False)

        header = tk.Frame(box, bg="white")
        header.pack(fill="x", padx=10, pady=(4, 0))

        tk.Label(
            header,
            text="THRESHOLD",
            bg="white",
            fg="#666",
            font=("Segoe UI", 7),
            anchor="w",
        ).pack(side="left")

        tk.Label(
            header,
            textvariable=self.threshold_label_var,
            bg="white",
            fg="#111",
            font=("Segoe UI", 9, "bold"),
            anchor="e",
        ).pack(side="right")

        slider = tk.Scale(
            box,
            variable=self.threshold_var,
            from_=0,
            to=100,
            orient="horizontal",
            resolution=1,
            showvalue=False,
            command=self.on_threshold_change,
            bg="white",
            troughcolor="#dddddd",
            highlightthickness=0,
            length=self.panel_width - 28,
        )
        slider.pack(fill="x", padx=8, pady=(0, 0))

    def on_threshold_change(self, value: str) -> None:
        self.threshold_label_var.set(f"{float(value):.0f}%")

    def make_last_accepted_row(self) -> None:
        row_height = 84
        row = tk.Frame(self.side_panel, width=self.panel_width, height=row_height, bg="#f4f4f4")
        row.pack(fill="x", pady=(0, 6))
        row.pack_propagate(False)

        accepted_cell = tk.Frame(row, width=300, height=row_height, bg="#f4f4f4")
        accepted_cell.pack(side="left", padx=(0, 8))
        accepted_cell.pack_propagate(False)
        self.make_info_box(
            "Last Accepted",
            self.last_positive_var,
            height=row_height,
            font=("Segoe UI", 10, "bold"),
            parent=accepted_cell,
            width=300,
            wraplength=278,
        )

        lights_cell = tk.Frame(row, width=112, height=row_height, bg="#f4f4f4")
        lights_cell.pack(side="left")
        lights_cell.pack_propagate(False)
        self.make_lights_box(lights_cell, width=112, height=row_height)

    def make_lights_box(self, parent: tk.Widget, width: int, height: int) -> None:
        box = tk.Frame(
            parent,
            width=width,
            height=height,
            bg="white",
            highlightbackground="#d0d0d0",
            highlightthickness=1,
        )
        box.pack(fill="both", expand=True)
        box.pack_propagate(False)

        tk.Label(
            box,
            text="LIGHTS",
            bg="white",
            fg="#666",
            font=("Segoe UI", 7),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(4, 0))

        grid = tk.Frame(box, bg="white", width=76, height=54)
        grid.pack(padx=10, pady=(1, 6))
        grid.grid_propagate(False)

        positions = {
            "up": (0, 1),
            "left": (1, 0),
            "right": (1, 2),
            "down": (2, 1),
        }

        for row in range(3):
            grid.grid_rowconfigure(row, minsize=18)
        for column in range(3):
            grid.grid_columnconfigure(column, minsize=24)

        for direction, (row, column) in positions.items():
            light = tk.Frame(
                grid,
                width=16,
                height=16,
                bg=LIGHT_OFF_COLOR,
                highlightbackground="#777",
                highlightthickness=1,
            )
            light.grid(row=row, column=column, padx=4, pady=1)
            light.grid_propagate(False)
            self.light_boxes[direction] = light

    def set_active_light(self, active_direction: str | None) -> None:
        for direction, light in self.light_boxes.items():
            color = LIGHT_ON_COLOR if direction == active_direction else LIGHT_OFF_COLOR
            light.configure(bg=color)

    def update_demo_lights(self, label: str) -> None:
        if label == "Thumb Down":
            self.set_active_light(None)
            return

        direction = DIRECTION_COMMANDS.get(label)
        if direction is not None:
            self.set_active_light(direction)

    def start(self) -> None:
        self.reader.start()
        self.root.after(20, self.poll_queue)
        self.root.mainloop()

    def close(self) -> None:
        self.stop_event.set()
        self.root.after(100, self.root.destroy)

    def poll_queue(self) -> None:
        while True:
            try:
                item = self.frame_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(item, ReaderStatus):
                self.status_var.set(item.message)
                self.status_label.configure(fg="#b00020" if item.is_error else "#111")
            else:
                self.update_packet(item)

        if not self.stop_event.is_set():
            self.root.after(20, self.poll_queue)

    def update_packet(self, packet: FramePacket) -> None:
        if packet.image is not None:
            image = Image.fromarray(packet.image, mode="RGB")
            if self.args.scale != 1:
                image = image.resize(
                    (packet.width * self.args.scale, packet.height * self.args.scale),
                    Image.Resampling.NEAREST,
                )

            self.current_photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.current_photo)

        label, pred_class, confidence, confidence_value, metadata = parse_result_fields(packet.result)
        cnn_us = metadata.get("cnn_us")
        profile = parse_profile_metadata(metadata)
        frame = metadata.get("frame", "--")
        policy = metadata.get("policy", "--")
        prefix = metadata.get("prefix", "--")
        accepted = metadata.get("accepted") == "1"
        skipped = metadata.get("skipped") == "1" or cnn_us == "0"
        ran_cnn = cnn_us is not None and cnn_us.isdigit() and int(cnn_us) > 0

        self.packet_count += 1
        if ran_cnn:
            self.inference_count += 1
        if packet.has_image:
            self.display_count += 1

        elapsed = max(time.monotonic() - self.start_time, 1e-6)
        inference_fps = self.inference_count / elapsed
        display_fps = self.display_count / elapsed

        self.prediction_var.set(label)
        self.class_var.set(pred_class)
        self.confidence_var.set(confidence)
        if (
            (accepted or (confidence_value is not None and confidence_value >= self.threshold_var.get()))
            and pred_class not in NEGATIVE_CLASSES
        ):
            self.last_positive_var.set(f"{label}\nclass {pred_class} | {confidence} | frame {frame}")
            self.update_demo_lights(label)
        self.inference_fps_var.set(f"{inference_fps:.2f}")
        self.display_fps_var.set(f"{display_fps:.2f}")
        if ran_cnn and cnn_us is not None:
            self.cnn_time_var.set(f"{int(cnn_us) / 1000:.2f} ms")
        elif skipped:
            self.cnn_time_var.set("skip")
        else:
            self.cnn_time_var.set("--")

        if self.args.profile:
            self.update_profile_boxes(profile)

        if packet.has_image:
            self.packet_var.set(f"#{frame} img p{prefix} {policy}")
            self.status_var.set(f"{packet.width}x{packet.height} {packet.pixel_format}")
        else:
            self.packet_var.set(f"#{frame} result p{prefix} {policy}")
            self.status_var.set("accepted" if accepted else ("CNN skipped" if skipped else "CNN run"))
        self.status_label.configure(fg="#111")

    def update_profile_boxes(self, profile: dict[str, int]) -> None:
        if not profile:
            for variable in (
                self.capture_time_var,
                self.preprocess_time_var,
                self.load_time_var,
                self.softmax_time_var,
                self.total_time_var,
                self.cycle_count_var,
            ):
                variable.set("--")
            return

        mhz = max(profile.get("mhz", 0), 1)
        preproc_us = profile["get"] + profile["history"]
        total_cycles = profile["total"] * mhz
        cnn_cycles = profile["cnn"] * mhz
        self.capture_time_var.set(f"{profile['capture'] / 1000:.2f} ms")
        self.preprocess_time_var.set(f"{preproc_us / 1000:.2f} ms")
        self.load_time_var.set(f"{profile['load'] / 1000:.2f} ms")
        self.softmax_time_var.set(f"{profile['softmax'] / 1000:.2f} ms")
        self.total_time_var.set(f"{profile['total'] / 1000:.2f} ms")
        self.cycle_count_var.set(f"T {total_cycles:,}\nCNN {cnn_cycles:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tk realtime MAX78000 camera viewer")
    parser.add_argument("--port", default="COM5", help="serial port connected to the MAX78000")
    parser.add_argument("--baud", type=int, default=2000000, help="serial baud rate")
    parser.add_argument("--timeout", type=float, default=2.0, help="serial read timeout in seconds")
    parser.add_argument("--scale", type=int, default=6, help="integer display scale for the 64x64 frame")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD_PERCENT,
        help="confidence threshold percentage for the last accepted gesture box",
    )
    parser.add_argument("--profile", action="store_true", help="show firmware timing and approximate cycle counts")
    return parser.parse_args()


if __name__ == "__main__":
    CameraApp(parse_args()).start()
