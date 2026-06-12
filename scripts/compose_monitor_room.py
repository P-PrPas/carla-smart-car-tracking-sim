#!/usr/bin/env python3
"""Compose the seven Honda CARLA CCTV videos into one monitor-room view."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CAMERA_IDS = [
    "CAM_01_START",
    "CAM_02_TRANSIT",
    "CAM_03_JUNCTION_STATUS",
    "CAM_04_GOOD_ROUTE",
    "CAM_05_DEFECT_ROUTE",
    "CAM_06_GOOD_PARKING",
    "CAM_07_DEFECT_PARKING",
]

CAMERA_NAMES = {
    "CAM_01_START": "CAM 01  START",
    "CAM_02_TRANSIT": "CAM 02  TRANSIT",
    "CAM_03_JUNCTION_STATUS": "CAM 03  JUNCTION",
    "CAM_04_GOOD_ROUTE": "CAM 04  GOOD ROUTE",
    "CAM_05_DEFECT_ROUTE": "CAM 05  DEFECT ROUTE",
    "CAM_06_GOOD_PARKING": "CAM 06  GOOD PARKING",
    "CAM_07_DEFECT_PARKING": "CAM 07  DEFECT PARKING",
}

CAMERA_SUBTITLES = {
    "CAM_01_START": "vehicle entry",
    "CAM_02_TRANSIT": "same road after start",
    "CAM_03_JUNCTION_STATUS": "branch decision",
    "CAM_04_GOOD_ROUTE": "good vehicle lane",
    "CAM_05_DEFECT_ROUTE": "defect vehicle lane",
    "CAM_06_GOOD_PARKING": "good parking slots",
    "CAM_07_DEFECT_PARKING": "defect parking slots",
}

CAMERA_COLORS = {
    "CAM_01_START": (210, 210, 210),
    "CAM_02_TRANSIT": (235, 210, 120),
    "CAM_03_JUNCTION_STATUS": (90, 190, 255),
    "CAM_04_GOOD_ROUTE": (90, 225, 145),
    "CAM_05_DEFECT_ROUTE": (105, 130, 255),
    "CAM_06_GOOD_PARKING": (65, 235, 145),
    "CAM_07_DEFECT_PARKING": (95, 95, 255),
}


@dataclass(frozen=True)
class TileSpec:
    camera_id: str
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def left(self) -> tuple[int, int]:
        return self.x, self.y + self.h // 2

    @property
    def right(self) -> tuple[int, int]:
        return self.x + self.w, self.y + self.h // 2

    @property
    def top(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y

    @property
    def bottom(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h


LAYOUT_1080P = {
    "CAM_01_START": TileSpec("CAM_01_START", 48, 128, 420, 236),
    "CAM_02_TRANSIT": TileSpec("CAM_02_TRANSIT", 48, 606, 420, 236),
    "CAM_03_JUNCTION_STATUS": TileSpec("CAM_03_JUNCTION_STATUS", 552, 360, 500, 281),
    "CAM_04_GOOD_ROUTE": TileSpec("CAM_04_GOOD_ROUTE", 1132, 128, 342, 192),
    "CAM_06_GOOD_PARKING": TileSpec("CAM_06_GOOD_PARKING", 1530, 128, 342, 192),
    "CAM_05_DEFECT_ROUTE": TileSpec("CAM_05_DEFECT_ROUTE", 1132, 650, 342, 192),
    "CAM_07_DEFECT_PARKING": TileSpec("CAM_07_DEFECT_PARKING", 1530, 650, 342, 192),
}

FLOW_EDGES = [
    ("CAM_01_START", "CAM_02_TRANSIT", "START"),
    ("CAM_02_TRANSIT", "CAM_03_JUNCTION_STATUS", "TRANSIT"),
    ("CAM_03_JUNCTION_STATUS", "CAM_04_GOOD_ROUTE", "GOOD"),
    ("CAM_04_GOOD_ROUTE", "CAM_06_GOOD_PARKING", "PARK"),
    ("CAM_03_JUNCTION_STATUS", "CAM_05_DEFECT_ROUTE", "DEFECT"),
    ("CAM_05_DEFECT_ROUTE", "CAM_07_DEFECT_PARKING", "PARK"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose seven CCTV camera videos into one monitor-room MP4.")
    parser.add_argument("--input-dir", default="datasets/carla_honda_poc/videos")
    parser.add_argument("--output", default="datasets/carla_honda_poc/videos/monitor_room.mp4")
    parser.add_argument("--metadata-dir", default="datasets/carla_honda_poc/metadata")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS. Default uses the first input video FPS.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap for quick previews.")
    parser.add_argument("--show-frame-id", action="store_true")
    parser.add_argument("--codec", default="mp4v")
    return parser.parse_args()


def require_1080p_layout(width: int, height: int) -> None:
    if (width, height) != (1920, 1080):
        raise ValueError("This monitor-room layout is currently tuned for --width 1920 --height 1080.")


def open_captures(input_dir: Path) -> dict[str, cv2.VideoCapture]:
    captures: dict[str, cv2.VideoCapture] = {}
    missing: list[str] = []
    for camera_id in CAMERA_IDS:
        path = input_dir / f"{camera_id}.mp4"
        if not path.exists():
            missing.append(str(path))
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {path}")
        captures[camera_id] = cap
    if missing:
        raise FileNotFoundError("Missing camera videos:\n" + "\n".join(missing))
    return captures


def input_video_info(captures: dict[str, cv2.VideoCapture]) -> tuple[float, int]:
    fps_values = []
    frame_counts = []
    for camera_id, cap in captures.items():
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise RuntimeError(f"Input video {camera_id} has invalid FPS: {fps}")
        if count <= 0:
            raise RuntimeError(f"Input video {camera_id} has no frames.")
        fps_values.append(fps)
        frame_counts.append(count)
    return fps_values[0], min(frame_counts)


def read_camera_graph(metadata_dir: Path) -> dict[str, object]:
    path = metadata_dir / "camera_graph.json"
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def rounded_rect(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
    thickness: int = -1,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius), (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(image, (cx, cy), radius, color, -1)
    else:
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def text_size(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    (w, h), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h


def draw_background(width: int, height: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top = np.array([26, 29, 31], dtype=np.float32)
    bottom = np.array([14, 16, 18], dtype=np.float32)
    row = top * (1.0 - y) + bottom * y
    image = np.repeat(row[:, None, :], width, axis=1).astype(np.uint8)

    for x in range(0, width, 48):
        cv2.line(image, (x, 88), (x, height - 58), (35, 38, 40), 1, cv2.LINE_AA)
    for y_pos in range(112, height - 58, 48):
        cv2.line(image, (24, y_pos), (width - 24, y_pos), (35, 38, 40), 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (width, 88), (20, 22, 24), -1)
    cv2.rectangle(image, (0, height - 56), (width, height), (20, 22, 24), -1)
    return image


def draw_header(canvas: np.ndarray, frame_idx: int, fps: float, metadata: dict[str, object], show_frame_id: bool) -> None:
    title = "HONDA SMART CAR TRACKING  |  CCTV MONITOR ROOM"
    put_text(canvas, title, (42, 42), 0.78, (235, 238, 238), 2)
    subtitle = "Flow: CAM01 -> CAM02 -> CAM03 -> GOOD(CAM04->CAM06) / DEFECT(CAM05->CAM07)"
    put_text(canvas, subtitle, (42, 70), 0.46, (150, 162, 162), 1)

    timestamp = frame_idx / fps if fps else 0.0
    label = f"T+{timestamp:05.1f}s"
    if show_frame_id:
        label += f"   FRAME {frame_idx:05d}"
    w, _ = text_size(label, 0.62, 2)
    rounded_rect(canvas, (canvas.shape[1] - w - 76, 24), (canvas.shape[1] - 40, 64), (42, 48, 50), 8)
    put_text(canvas, label, (canvas.shape[1] - w - 58, 51), 0.62, (235, 238, 238), 2)

    carla_map = metadata.get("carla_map") or metadata.get("map") or ""
    if isinstance(metadata.get("cameras"), list):
        carla_map = metadata.get("carla_map", carla_map)
    if carla_map:
        put_text(canvas, str(carla_map), (canvas.shape[1] - 320, 78), 0.42, (130, 140, 140), 1)


def draw_arrow(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    sx, sy = start
    ex, ey = end
    mid = ((sx + ex) // 2, (sy + ey) // 2)
    cv2.arrowedLine(canvas, start, end, color, 3, cv2.LINE_AA, tipLength=0.08)
    tw, th = text_size(label, 0.42, 1)
    pad_x, pad_y = 9, 7
    box1 = (mid[0] - tw // 2 - pad_x, mid[1] - th // 2 - pad_y)
    box2 = (mid[0] + tw // 2 + pad_x, mid[1] + th // 2 + pad_y)
    rounded_rect(canvas, box1, box2, (30, 33, 34), 7)
    rounded_rect(canvas, box1, box2, color, 7, 1)
    put_text(canvas, label, (mid[0] - tw // 2, mid[1] + th // 2 - 1), 0.42, (232, 235, 232), 1)


def draw_flow(canvas: np.ndarray) -> None:
    layout = LAYOUT_1080P
    edge_points = {
        ("CAM_01_START", "CAM_02_TRANSIT"): (layout["CAM_01_START"].bottom, layout["CAM_02_TRANSIT"].top),
        ("CAM_02_TRANSIT", "CAM_03_JUNCTION_STATUS"): (layout["CAM_02_TRANSIT"].right, layout["CAM_03_JUNCTION_STATUS"].left),
        ("CAM_03_JUNCTION_STATUS", "CAM_04_GOOD_ROUTE"): (layout["CAM_03_JUNCTION_STATUS"].right, layout["CAM_04_GOOD_ROUTE"].left),
        ("CAM_04_GOOD_ROUTE", "CAM_06_GOOD_PARKING"): (layout["CAM_04_GOOD_ROUTE"].right, layout["CAM_06_GOOD_PARKING"].left),
        ("CAM_03_JUNCTION_STATUS", "CAM_05_DEFECT_ROUTE"): (layout["CAM_03_JUNCTION_STATUS"].right, layout["CAM_05_DEFECT_ROUTE"].left),
        ("CAM_05_DEFECT_ROUTE", "CAM_07_DEFECT_PARKING"): (layout["CAM_05_DEFECT_ROUTE"].right, layout["CAM_07_DEFECT_PARKING"].left),
    }
    for src, dst, label in FLOW_EDGES:
        color = (100, 220, 145) if label in {"GOOD", "PARK"} and "GOOD" in dst else CAMERA_COLORS.get(dst, (210, 210, 210))
        if label == "DEFECT" or "DEFECT" in dst:
            color = (105, 130, 255)
        draw_arrow(canvas, *edge_points[(src, dst)], label, color)


def cover_resize(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(target_w / w, target_h / h)
    resized_w = int(math.ceil(w * scale))
    resized_h = int(math.ceil(h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x0 = max(0, (resized_w - target_w) // 2)
    y0 = max(0, (resized_h - target_h) // 2)
    return resized[y0 : y0 + target_h, x0 : x0 + target_w]


def draw_tile(canvas: np.ndarray, spec: TileSpec, frame: np.ndarray, frame_idx: int) -> None:
    color = CAMERA_COLORS[spec.camera_id]
    shadow = np.zeros_like(canvas)
    rounded_rect(shadow, (spec.x - 5, spec.y - 5), (spec.x + spec.w + 5, spec.y + spec.h + 38), (0, 0, 0), 10)
    cv2.addWeighted(shadow, 0.22, canvas, 1.0, 0.0, canvas)

    rounded_rect(canvas, (spec.x - 3, spec.y - 3), (spec.x + spec.w + 3, spec.y + spec.h + 34), (35, 38, 40), 9)
    rounded_rect(canvas, (spec.x - 3, spec.y - 3), (spec.x + spec.w + 3, spec.y + spec.h + 34), color, 9, 2)

    image = cover_resize(frame, spec.w, spec.h)
    canvas[spec.y : spec.y + spec.h, spec.x : spec.x + spec.w] = image
    cv2.rectangle(canvas, (spec.x, spec.y), (spec.x + spec.w, spec.y + spec.h), color, 2, cv2.LINE_AA)

    header_h = 34
    overlay = canvas.copy()
    cv2.rectangle(overlay, (spec.x, spec.y), (spec.x + spec.w, spec.y + header_h), (8, 10, 11), -1)
    cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0.0, canvas)
    cv2.circle(canvas, (spec.x + 18, spec.y + 17), 6, (68, 238, 125), -1, cv2.LINE_AA)
    put_text(canvas, CAMERA_NAMES[spec.camera_id], (spec.x + 34, spec.y + 23), 0.5, (246, 248, 248), 1)
    put_text(canvas, "LIVE", (spec.x + spec.w - 52, spec.y + 23), 0.42, (68, 238, 125), 1)

    footer_y = spec.y + spec.h + 23
    put_text(canvas, CAMERA_SUBTITLES[spec.camera_id], (spec.x + 10, footer_y), 0.45, (180, 188, 188), 1)
    put_text(canvas, f"f={frame_idx:05d}", (spec.x + spec.w - 96, footer_y), 0.42, (120, 130, 130), 1)


def draw_footer(canvas: np.ndarray) -> None:
    y = canvas.shape[0] - 22
    put_text(canvas, "monitor layout preserves route continuity; arrows indicate expected cross-camera handoff", (42, y), 0.45, (130, 140, 140), 1)
    put_text(canvas, "GOOD branch", (1228, y), 0.45, (95, 230, 145), 1)
    put_text(canvas, "DEFECT branch", (1438, y), 0.45, (110, 135, 255), 1)
    put_text(canvas, "PARKING", (1670, y), 0.45, (225, 225, 150), 1)


def compose_frame(
    frames: dict[str, np.ndarray],
    width: int,
    height: int,
    frame_idx: int,
    fps: float,
    metadata: dict[str, object],
    show_frame_id: bool,
) -> np.ndarray:
    canvas = draw_background(width, height)
    draw_header(canvas, frame_idx, fps, metadata, show_frame_id)
    draw_flow(canvas)
    for camera_id in CAMERA_IDS:
        draw_tile(canvas, LAYOUT_1080P[camera_id], frames[camera_id], frame_idx)
    draw_footer(canvas)
    return canvas


def main() -> int:
    args = parse_args()
    require_1080p_layout(args.width, args.height)

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    captures = open_captures(input_dir)
    metadata = read_camera_graph(Path(args.metadata_dir))

    first_fps, frame_count = input_video_info(captures)
    output_fps = args.fps if args.fps > 0 else first_fps
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)

    fourcc = cv2.VideoWriter_fourcc(*args.codec[:4])
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {output_path}")

    try:
        for frame_idx in range(frame_count):
            frames: dict[str, np.ndarray] = {}
            for camera_id, cap in captures.items():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Could not read frame {frame_idx} from {camera_id}.")
                frames[camera_id] = frame
            canvas = compose_frame(frames, args.width, args.height, frame_idx, output_fps, metadata, args.show_frame_id)
            writer.write(canvas)
            if frame_idx == 0 or (frame_idx + 1) % max(1, int(output_fps) * 5) == 0 or frame_idx + 1 == frame_count:
                pct = (frame_idx + 1) / frame_count * 100.0
                print(f"[monitor] frame {frame_idx + 1}/{frame_count} ({pct:5.1f}%)")
    finally:
        writer.release()
        for cap in captures.values():
            cap.release()

    print(f"Wrote monitor-room video: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
