#!/usr/bin/env python3
"""Compose the seven Honda CARLA CCTV videos into one monitor-room view."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
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


@dataclass(frozen=True)
class MonitorConfig:
    show_bboxes: bool = False


LAYOUT_1080P = {
    "CAM_02_TRANSIT": TileSpec("CAM_02_TRANSIT", 72, 170, 344, 194),
    "CAM_01_START": TileSpec("CAM_01_START", 218, 626, 344, 194),
    "CAM_03_JUNCTION_STATUS": TileSpec("CAM_03_JUNCTION_STATUS", 730, 380, 374, 210),
    "CAM_05_DEFECT_ROUTE": TileSpec("CAM_05_DEFECT_ROUTE", 1198, 116, 344, 194),
    "CAM_07_DEFECT_PARKING": TileSpec("CAM_07_DEFECT_PARKING", 1530, 118, 344, 194),
    "CAM_04_GOOD_ROUTE": TileSpec("CAM_04_GOOD_ROUTE", 1180, 696, 344, 194),
    "CAM_06_GOOD_PARKING": TileSpec("CAM_06_GOOD_PARKING", 1530, 690, 344, 194),
}

MAP_POINTS = {
    "CAM_01_START": (168, 720),
    "CAM_02_TRANSIT": (505, 426),
    "CAM_03_JUNCTION_STATUS": (855, 640),
    "CAM_04_GOOD_ROUTE": (1115, 792),
    "CAM_06_GOOD_PARKING": (1658, 875),
    "CAM_05_DEFECT_ROUTE": (1168, 275),
    "CAM_07_DEFECT_PARKING": (1530, 418),
}

ROUTE_PATHS = {
    "common": [
        MAP_POINTS["CAM_01_START"],
        (255, 672),
        (340, 545),
        MAP_POINTS["CAM_02_TRANSIT"],
        (626, 454),
        (735, 552),
        MAP_POINTS["CAM_03_JUNCTION_STATUS"],
    ],
    "good": [
        MAP_POINTS["CAM_03_JUNCTION_STATUS"],
        MAP_POINTS["CAM_04_GOOD_ROUTE"],
        (1325, 878),
        MAP_POINTS["CAM_06_GOOD_PARKING"],
    ],
    "defect": [
        MAP_POINTS["CAM_03_JUNCTION_STATUS"],
        (1000, 520),
        (1048, 368),
        MAP_POINTS["CAM_05_DEFECT_ROUTE"],
        (1320, 482),
        MAP_POINTS["CAM_07_DEFECT_PARKING"],
    ],
}

ROUTE_COLORS = {
    "common": (255, 142, 46),
    "good": (88, 255, 114),
    "defect": (72, 82, 255),
}

NEUTRAL_BBOX_COLOR = (235, 210, 120)
CLASSIFICATION_CAMERA_ID = "CAM_03_JUNCTION_STATUS"
PRE_CLASSIFICATION_CAMERAS = {"CAM_01_START", "CAM_02_TRANSIT"}
POST_CLASSIFICATION_CAMERAS = {
    "CAM_04_GOOD_ROUTE",
    "CAM_05_DEFECT_ROUTE",
    "CAM_06_GOOD_PARKING",
    "CAM_07_DEFECT_PARKING",
}

CAM3_BASE_SIZE = (374, 210)
CAM3_GOOD_GATE = [(76, 166), (368, 150), (373, 209), (66, 209)]
CAM3_DEFECT_GATE = [(0, 38), (92, 50), (104, 158), (0, 192)]

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
    parser.add_argument("--input-dir", default="datasets/carla_honda_poc_aaa_relight/videos")
    parser.add_argument("--output", default="datasets/carla_honda_poc_aaa_relight/videos/visual_map_monitor.mp4")
    parser.add_argument("--metadata-dir", default="datasets/carla_honda_poc_aaa_relight/metadata")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS. Default uses the first input video FPS.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap for quick previews.")
    parser.add_argument("--show-frame-id", action="store_true")
    parser.add_argument("--bbox-mode", choices=("off", "on"), default="off", help="Draw vehicle tracking boxes from annotations.")
    parser.add_argument("--show-bboxes", action="store_true", help="Shortcut for --bbox-mode on.")
    parser.add_argument(
        "--annotations-path",
        default="",
        help="Path to bboxes.jsonl. Default: <input dataset>/annotations/bboxes.jsonl.",
    )
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


def default_annotations_path(input_dir: Path) -> Path:
    return input_dir.parent / "annotations" / "bboxes.jsonl"


def read_bbox_index(path: Path) -> dict[tuple[str, int], list[dict[str, object]]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find bbox annotations: {path}")

    index: defaultdict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                camera_id = str(record["camera_id"])
                frame_id = int(record["frame_id"])
                bbox = record["bbox"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid bbox annotation at {path}:{line_no}") from exc
            if camera_id not in CAMERA_IDS or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            index[(camera_id, frame_id)].append(record)
    return dict(index)


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


def blend_polygon(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], alpha: float) -> None:
    overlay = image.copy()
    cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)


def draw_rotated_rect(
    image: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    angle_deg: float,
    color: tuple[int, int, int],
    alpha: float,
    outline: tuple[int, int, int] | None = None,
) -> None:
    rect = (center, size, angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [box], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)
    if outline:
        cv2.polylines(image, [box], True, outline, 1, cv2.LINE_AA)


def draw_parking_block(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    slot_cols: int,
    slot_rows: int,
) -> None:
    x, y, w, h = rect
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (34, 42, 44), -1)
    cv2.addWeighted(overlay, 0.44, image, 0.56, 0.0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (78, 95, 98), 1, cv2.LINE_AA)
    cv2.rectangle(image, (x + 4, y + 4), (x + w - 4, y + h - 4), (44, 120, 135), 1, cv2.LINE_AA)

    margin_x = max(14, w // 12)
    margin_y = max(16, h // 12)
    cell_w = max(10, (w - margin_x * 2) // max(1, slot_cols))
    cell_h = max(8, (h - margin_y * 2) // max(1, slot_rows))
    slot_color = (76, 98, 100)
    car_color = (75, 74, 128)
    for row_idx in range(slot_rows):
        for col_idx in range(slot_cols):
            sx = x + margin_x + col_idx * cell_w
            sy = y + margin_y + row_idx * cell_h
            cv2.rectangle(image, (sx, sy), (sx + cell_w - 3, sy + cell_h - 3), slot_color, 1, cv2.LINE_AA)
            if (row_idx + col_idx) % 3 != 0:
                cv2.rectangle(
                    image,
                    (sx + 3, sy + 3),
                    (sx + cell_w - 6, sy + max(4, cell_h - 6)),
                    car_color,
                    -1,
                    cv2.LINE_AA,
                )

    tw, th = text_size(label, 1.25, 2)
    rounded_rect(
        image,
        (x + w // 2 - tw // 2 - 20, y + h // 2 - th // 2 - 18),
        (x + w // 2 + tw // 2 + 20, y + h // 2 + th // 2 + 18),
        (20, 23, 25),
        5,
    )
    put_text(image, label, (x + w // 2 - tw // 2, y + h // 2 + th // 2 - 1), 1.25, (232, 235, 232), 2)


def blend_polyline(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], thickness: int, alpha: float) -> None:
    overlay = image.copy()
    cv2.polylines(overlay, [np.array(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)


def draw_map_road(image: np.ndarray, points: list[tuple[int, int]]) -> None:
    curve = catmull_rom_path(points, samples_per_segment=40)
    blend_polyline(image, curve, (2, 4, 6), 68, 0.30)
    blend_polyline(image, curve, (42, 48, 49), 52, 0.72)
    blend_polyline(image, curve, (16, 21, 23), 40, 0.92)
    blend_polyline(image, curve, (88, 98, 98), 54, 0.12)

    arr = np.array(curve, dtype=np.int32)
    cv2.polylines(image, [arr], False, (80, 88, 88), 2, cv2.LINE_AA)
    cv2.polylines(image, [arr], False, (26, 34, 36), 34, cv2.LINE_AA)


def draw_zone_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    tw, th = text_size(text, 0.42, 1)
    rounded_rect(image, (x - 12, y - th - 11), (x + tw + 12, y + 10), (8, 12, 14), 5)
    cv2.rectangle(image, (x - 12, y + 12), (x + tw + 12, y + 14), color, -1)
    put_text(image, text, (x, y), 0.42, (126, 142, 142), 1)


def draw_map_lot(
    image: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    angle_deg: float,
    label: str,
    accent: tuple[int, int, int],
) -> None:
    cx, cy = center
    w, h = size
    draw_rotated_rect(image, center, size, angle_deg, (38, 45, 47), 0.58, (74, 88, 88))
    draw_rotated_rect(image, center, (max(20, w - 18), max(20, h - 18)), angle_deg, (22, 27, 29), 0.30, accent)

    angle = math.radians(angle_deg)
    right = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    down = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
    cols = max(3, w // 36)
    rows = max(2, h // 30)
    start = np.array([cx, cy], dtype=np.float32) - right * (cols - 1) * 18 - down * (rows - 1) * 15
    for row_idx in range(rows):
        for col_idx in range(cols):
            p = start + right * col_idx * 36 + down * row_idx * 30
            draw_rotated_rect(image, (int(p[0]), int(p[1])), (24, 10), angle_deg, (84, 96, 98), 0.34)

    tw, th = text_size(label, 0.48, 1)
    draw_rotated_rect(image, (cx, cy), (tw + 42, th + 34), angle_deg, (10, 14, 16), 0.78, accent)
    put_text(image, label, (cx - tw // 2, cy + th // 2), 0.48, (174, 188, 188), 1)


def catmull_rom_path(points: list[tuple[int, int]], samples_per_segment: int = 28) -> list[tuple[int, int]]:
    if len(points) < 2:
        return points
    pts = [points[0], *points, points[-1]]
    curve: list[tuple[int, int]] = []
    for idx in range(1, len(pts) - 2):
        p0 = np.array(pts[idx - 1], dtype=np.float32)
        p1 = np.array(pts[idx], dtype=np.float32)
        p2 = np.array(pts[idx + 1], dtype=np.float32)
        p3 = np.array(pts[idx + 2], dtype=np.float32)
        for step in range(samples_per_segment):
            t = step / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            curve.append((int(round(point[0])), int(round(point[1]))))
    curve.append(points[-1])
    return curve


def draw_polyline_glow(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    arr = np.array(points, dtype=np.int32)
    for thickness, alpha in ((34, 0.08), (22, 0.12), (12, 0.20)):
        overlay = image.copy()
        cv2.polylines(overlay, [arr], False, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)
    cv2.polylines(image, [arr], False, color, 6, cv2.LINE_AA)
    cv2.polylines(image, [arr], False, (245, 250, 255), 1, cv2.LINE_AA)


def draw_dashed_motion(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], frame_idx: int) -> None:
    if len(points) < 2:
        return
    dash = 28
    gap = 30
    offset = (frame_idx * 3) % (dash + gap)
    flat: list[tuple[np.ndarray, float]] = []
    cumulative = 0.0
    prev = np.array(points[0], dtype=np.float32)
    flat.append((prev, cumulative))
    for point in points[1:]:
        current = np.array(point, dtype=np.float32)
        cumulative += float(np.linalg.norm(current - prev))
        flat.append((current, cumulative))
        prev = current

    total = cumulative
    if total <= 0:
        return

    def interp(distance: float) -> tuple[int, int]:
        distance = max(0.0, min(total, distance))
        for idx in range(1, len(flat)):
            p0, d0 = flat[idx - 1]
            p1, d1 = flat[idx]
            if d1 >= distance:
                ratio = 0.0 if d1 == d0 else (distance - d0) / (d1 - d0)
                p = p0 + (p1 - p0) * ratio
                return int(round(p[0])), int(round(p[1]))
        p = flat[-1][0]
        return int(round(p[0])), int(round(p[1]))

    distance = -offset
    while distance < total:
        start = interp(distance)
        end = interp(distance + dash)
        cv2.line(image, start, end, (245, 250, 255), 2, cv2.LINE_AA)
        distance += dash + gap

    for marker_distance in np.arange(90 + offset, total, 190):
        tip = interp(float(marker_distance))
        tail = interp(float(marker_distance - 18))
        cv2.arrowedLine(image, tail, tip, color, 2, cv2.LINE_AA, tipLength=0.55)


@lru_cache(maxsize=4)
def build_background(width: int, height: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    top = np.array([18, 27, 30], dtype=np.float32)
    bottom = np.array([5, 8, 11], dtype=np.float32)
    row = top * (1.0 - y) + bottom * y
    image = np.repeat(row[:, None, :], width, axis=1).astype(np.uint8)

    radial = np.sqrt((x - 0.52) ** 2 + (y - 0.50) ** 2)
    vignette = np.clip((radial - 0.16) / 0.78, 0.0, 1.0)
    image = np.clip(image.astype(np.float32) * (1.0 - 0.42 * vignette[:, :, None]), 0, 255).astype(np.uint8)

    rng = np.random.default_rng(12)
    noise = rng.normal(0, 4, (height, width, 1)).astype(np.float32)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    blend_polygon(image, [(0, 120), (430, 104), (250, 1080), (0, 1080)], (12, 50, 34), 0.32)
    blend_polygon(image, [(1260, 86), (1920, 72), (1920, 1080), (1440, 1020)], (8, 46, 32), 0.30)
    blend_polygon(image, [(515, 785), (735, 760), (850, 1045), (595, 1080)], (22, 42, 47), 0.38)
    blend_polygon(image, [(1150, 598), (1775, 570), (1920, 808), (1205, 840)], (15, 35, 42), 0.30)

    for route_name in ("common", "defect", "good"):
        draw_map_road(image, ROUTE_PATHS[route_name])

    draw_rotated_rect(image, (720, 260), (555, 230), -12, (50, 57, 59), 0.64, (82, 94, 94))
    draw_rotated_rect(image, (915, 295), (350, 110), -12, (63, 70, 72), 0.46, (92, 104, 104))
    draw_rotated_rect(image, (585, 665), (520, 160), -7, (48, 55, 57), 0.58, (78, 90, 90))
    draw_rotated_rect(image, (1390, 560), (460, 150), -15, (47, 54, 57), 0.46, (76, 90, 90))
    draw_rotated_rect(image, (1490, 946), (520, 115), 5, (38, 46, 50), 0.42, (68, 82, 84))

    draw_map_lot(image, (1456, 390), (260, 178), -3, "HOLDING AREA", ROUTE_COLORS["defect"])
    draw_map_lot(image, (1410, 826), (290, 172), 5, "GOOD STORAGE", ROUTE_COLORS["good"])

    draw_zone_label(image, "ENTRY ROAD", (180, 620), ROUTE_COLORS["common"])
    draw_zone_label(image, "TRANSIT CURVE", (440, 350), ROUTE_COLORS["common"])
    draw_zone_label(image, "INSPECTION JUNCTION", (760, 326), CAMERA_COLORS["CAM_03_JUNCTION_STATUS"])
    draw_zone_label(image, "GOOD FLOW", (1018, 882), ROUTE_COLORS["good"])
    draw_zone_label(image, "HOLD FLOW", (1035, 242), ROUTE_COLORS["defect"])

    for lx, ly in MAP_POINTS.values():
        overlay = image.copy()
        cv2.circle(overlay, (lx, ly), 86, (112, 126, 126), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.045, image, 0.955, 0.0, image)

    for x_pos in range(-80, width + 120, 84):
        cv2.line(image, (x_pos, 80), (x_pos - 260, height), (25, 34, 36), 1, cv2.LINE_AA)
    for y_pos in range(120, height + 60, 76):
        cv2.line(image, (0, y_pos), (width, y_pos - 210), (22, 31, 33), 1, cv2.LINE_AA)

    cv2.rectangle(image, (0, 0), (width, 56), (6, 8, 12), -1)
    cv2.line(image, (0, 56), (width, 56), (35, 44, 48), 1, cv2.LINE_AA)

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 56), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.20, image, 0.80, 0.0, image)
    return image


def draw_background(width: int, height: int) -> np.ndarray:
    return build_background(width, height).copy()


def draw_header(canvas: np.ndarray, frame_idx: int, fps: float, metadata: dict[str, object], show_frame_id: bool) -> None:
    put_text(canvas, "HONDA", (28, 38), 0.78, (38, 58, 245), 2)
    cv2.line(canvas, (178, 16), (178, 42), (70, 78, 84), 1, cv2.LINE_AA)
    put_text(canvas, "Honda Vehicle Visual Map Monitor", (198, 38), 0.64, (238, 242, 242), 1)

    timestamp = frame_idx / fps if fps else 0.0
    label = f"T+{timestamp:05.1f}s"
    if show_frame_id:
        label += f"   FRAME {frame_idx:05d}"
    put_text(canvas, "LIVE", (canvas.shape[1] - 344, 36), 0.48, (45, 65, 255), 2)
    cv2.circle(canvas, (canvas.shape[1] - 360, 28), 6, (45, 65, 255), -1, cv2.LINE_AA)
    w, _ = text_size(label, 0.48, 1)
    put_text(canvas, label, (canvas.shape[1] - w - 30, 36), 0.48, (210, 214, 214), 1)

    carla_map = metadata.get("carla_map") or metadata.get("map") or ""
    if isinstance(metadata.get("cameras"), list):
        carla_map = metadata.get("carla_map", carla_map)
    if carla_map:
        put_text(canvas, str(carla_map), (30, 82), 0.36, (108, 124, 126), 1)


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


def marker_label(camera_id: str) -> str:
    return {
        "CAM_01_START": "C01",
        "CAM_02_TRANSIT": "C02",
        "CAM_03_JUNCTION_STATUS": "C03",
        "CAM_04_GOOD_ROUTE": "C04",
        "CAM_05_DEFECT_ROUTE": "C05",
        "CAM_06_GOOD_PARKING": "C06",
        "CAM_07_DEFECT_PARKING": "C07",
    }[camera_id]


def draw_camera_marker(canvas: np.ndarray, camera_id: str, frame_idx: int) -> None:
    x, y = MAP_POINTS[camera_id]
    color = CAMERA_COLORS[camera_id]
    pulse = 1.0 + 0.08 * math.sin(frame_idx / 8.0)
    overlay = canvas.copy()
    cv2.circle(overlay, (x, y), int(25 * pulse), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0, canvas)

    label = marker_label(camera_id)
    tw, th = text_size(label, 0.44, 2)
    box1 = (x - tw // 2 - 21, y - th // 2 - 15)
    box2 = (x + tw // 2 + 21, y + th // 2 + 15)
    rounded_rect(canvas, box1, box2, (14, 22, 24), 6)
    rounded_rect(canvas, box1, box2, color, 6, 2)
    cv2.rectangle(canvas, (box1[0] + 9, y - 8), (box1[0] + 23, y + 6), color, -1)
    cv2.circle(canvas, (box1[0] + 26, y - 1), 5, color, -1, cv2.LINE_AA)
    put_text(canvas, label, (x - tw // 2 + 10, y + th // 2 - 1), 0.44, (246, 250, 248), 2)


def draw_leader_line(canvas: np.ndarray, camera_id: str) -> None:
    spec = LAYOUT_1080P[camera_id]
    marker = MAP_POINTS[camera_id]
    candidates = [spec.left, spec.right, spec.top, spec.bottom]
    anchor = min(candidates, key=lambda p: (p[0] - marker[0]) ** 2 + (p[1] - marker[1]) ** 2)
    color = CAMERA_COLORS[camera_id]
    overlay = canvas.copy()
    cv2.line(overlay, marker, anchor, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.38, canvas, 0.62, 0.0, canvas)


def draw_flow(canvas: np.ndarray, frame_idx: int) -> None:
    for route_name in ("common", "good", "defect"):
        curve = catmull_rom_path(ROUTE_PATHS[route_name])
        color = ROUTE_COLORS[route_name]
        draw_polyline_glow(canvas, curve, color)
        draw_dashed_motion(canvas, curve, color, frame_idx)
    for camera_id in CAMERA_IDS:
        draw_camera_marker(canvas, camera_id, frame_idx)
        draw_leader_line(canvas, camera_id)


def cover_resize(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(target_w / w, target_h / h)
    resized_w = int(math.ceil(w * scale))
    resized_h = int(math.ceil(h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x0 = max(0, (resized_w - target_w) // 2)
    y0 = max(0, (resized_h - target_h) // 2)
    return resized[y0 : y0 + target_h, x0 : x0 + target_w]


def map_bbox_to_tile(
    bbox: list[object],
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
) -> tuple[int, int, int, int] | None:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    scale = max(target_w / src_w, target_h / src_h)
    resized_w = int(math.ceil(src_w * scale))
    resized_h = int(math.ceil(src_h * scale))
    crop_x = max(0, (resized_w - target_w) // 2)
    crop_y = max(0, (resized_h - target_h) // 2)

    tx1 = int(round(x1 * scale - crop_x))
    ty1 = int(round(y1 * scale - crop_y))
    tx2 = int(round(x2 * scale - crop_x))
    ty2 = int(round(y2 * scale - crop_y))
    tx1 = max(0, min(target_w - 1, tx1))
    ty1 = max(0, min(target_h - 1, ty1))
    tx2 = max(0, min(target_w - 1, tx2))
    ty2 = max(0, min(target_h - 1, ty2))
    if tx2 - tx1 < 4 or ty2 - ty1 < 4:
        return None
    return tx1, ty1, tx2, ty2


def bbox_color(status: str | None) -> tuple[int, int, int]:
    if status == "GOOD":
        return ROUTE_COLORS["good"]
    if status == "DEFECT":
        return ROUTE_COLORS["defect"]
    return NEUTRAL_BBOX_COLOR


def short_tracking_label(record: dict[str, object], revealed_status: str | None) -> str:
    tracking_id = str(record.get("tracking_id", "TRK"))
    suffix = tracking_id.split("_")[-1]
    if suffix.isdigit():
        suffix = str(int(suffix))
    if revealed_status in {"GOOD", "DEFECT"}:
        return f"ID {suffix} {revealed_status}"
    return f"ID {suffix}"


def scaled_points(points: list[tuple[int, int]], target_w: int, target_h: int) -> list[tuple[int, int]]:
    base_w, base_h = CAM3_BASE_SIZE
    return [(int(round(x / base_w * target_w)), int(round(y / base_h * target_h))) for x, y in points]


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0


def revealed_tracking_status(
    camera_id: str,
    record: dict[str, object],
    bbox: tuple[int, int, int, int],
    target_w: int,
    target_h: int,
) -> str | None:
    status = str(record.get("status", "")).upper()
    if status not in {"GOOD", "DEFECT"}:
        return None
    if camera_id in PRE_CLASSIFICATION_CAMERAS:
        return None
    if camera_id in POST_CLASSIFICATION_CAMERAS:
        return status
    if camera_id != CLASSIFICATION_CAMERA_ID:
        return None

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    good_gate = scaled_points(CAM3_GOOD_GATE, target_w, target_h)
    defect_gate = scaled_points(CAM3_DEFECT_GATE, target_w, target_h)
    if status == "GOOD" and point_in_polygon((cx, cy), good_gate):
        return status
    if status == "DEFECT" and point_in_polygon((cx, cy), defect_gate):
        return status
    return None


def draw_cam3_decision_zones(image: np.ndarray) -> None:
    target_h, target_w = image.shape[:2]
    zones = [
        ("GOOD EXIT", scaled_points(CAM3_GOOD_GATE, target_w, target_h), ROUTE_COLORS["good"]),
        ("BAD EXIT", scaled_points(CAM3_DEFECT_GATE, target_w, target_h), ROUTE_COLORS["defect"]),
    ]
    for label, points, color in zones:
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, image, 0.82, 0.0, image)
        cv2.polylines(image, [np.array(points, dtype=np.int32)], True, color, 2, cv2.LINE_AA)
        cv2.polylines(image, [np.array(points, dtype=np.int32)], True, (245, 250, 248), 1, cv2.LINE_AA)

        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        text_x = max(4, min(target_w - 108, min_x + 6))
        text_y = max(18, min_y - 6)
        tw, th = text_size(label, 0.28, 1)
        rounded_rect(image, (text_x - 4, text_y - th - 6), (text_x + tw + 6, text_y + 4), (8, 12, 14), 4)
        rounded_rect(image, (text_x - 4, text_y - th - 6), (text_x + tw + 6, text_y + 4), color, 4, 1)
        put_text(image, label, (text_x, text_y - 2), 0.28, (235, 240, 240), 1)


def draw_tracking_bboxes(
    image: np.ndarray,
    camera_id: str,
    records: list[dict[str, object]],
    src_shape: tuple[int, int],
    config: MonitorConfig,
) -> None:
    src_h, src_w = src_shape
    target_h, target_w = image.shape[:2]
    for record in records:
        bbox = map_bbox_to_tile(record.get("bbox", []), src_w, src_h, target_w, target_h)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        revealed_status = revealed_tracking_status(camera_id, record, bbox, target_w, target_h)
        color = bbox_color(revealed_status)

        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.70, image, 0.30, 0.0, image)

        corner = max(9, min(22, (x2 - x1) // 4, (y2 - y1) // 4))
        for start, end in (
            ((x1, y1), (x1 + corner, y1)),
            ((x1, y1), (x1, y1 + corner)),
            ((x2, y1), (x2 - corner, y1)),
            ((x2, y1), (x2, y1 + corner)),
            ((x1, y2), (x1 + corner, y2)),
            ((x1, y2), (x1, y2 - corner)),
            ((x2, y2), (x2 - corner, y2)),
            ((x2, y2), (x2, y2 - corner)),
        ):
            cv2.line(image, start, end, (245, 250, 248), 1, cv2.LINE_AA)

        label = short_tracking_label(record, revealed_status)
        tw, th = text_size(label, 0.32, 1)
        label_x = max(0, min(target_w - tw - 10, x1))
        label_y = y1 - 7 if y1 >= th + 43 else min(target_h - 4, y2 + th + 10)
        box1 = (label_x, max(0, label_y - th - 8))
        box2 = (label_x + tw + 10, min(target_h - 1, label_y + 4))
        rounded_rect(image, box1, box2, (8, 12, 14), 4)
        rounded_rect(image, box1, box2, color, 4, 1)
        put_text(image, label, (label_x + 5, label_y - 2), 0.32, (235, 240, 240), 1)


def draw_tile(
    canvas: np.ndarray,
    spec: TileSpec,
    frame: np.ndarray,
    frame_idx: int,
    bbox_records: list[dict[str, object]],
    config: MonitorConfig,
) -> None:
    color = CAMERA_COLORS[spec.camera_id]
    shadow = np.zeros_like(canvas)
    rounded_rect(shadow, (spec.x - 8, spec.y - 8), (spec.x + spec.w + 8, spec.y + spec.h + 34), (0, 0, 0), 8)
    cv2.addWeighted(shadow, 0.34, canvas, 1.0, 0.0, canvas)

    rounded_rect(canvas, (spec.x - 3, spec.y - 3), (spec.x + spec.w + 3, spec.y + spec.h + 30), (10, 14, 17), 7)
    rounded_rect(canvas, (spec.x - 3, spec.y - 3), (spec.x + spec.w + 3, spec.y + spec.h + 30), (112, 128, 132), 7, 1)
    cv2.line(canvas, (spec.x, spec.y - 3), (spec.x + spec.w, spec.y - 3), color, 2, cv2.LINE_AA)

    image = cover_resize(frame, spec.w, spec.h)
    image = cv2.addWeighted(image, 0.92, np.zeros_like(image), 0.08, 0.0)
    if config.show_bboxes and spec.camera_id == CLASSIFICATION_CAMERA_ID:
        draw_cam3_decision_zones(image)
    if config.show_bboxes and bbox_records:
        draw_tracking_bboxes(image, spec.camera_id, bbox_records, frame.shape[:2], config)
    canvas[spec.y : spec.y + spec.h, spec.x : spec.x + spec.w] = image
    cv2.rectangle(canvas, (spec.x, spec.y), (spec.x + spec.w, spec.y + spec.h), (90, 105, 108), 1, cv2.LINE_AA)

    header_h = 30
    overlay = canvas.copy()
    cv2.rectangle(overlay, (spec.x, spec.y), (spec.x + spec.w, spec.y + header_h), (8, 10, 11), -1)
    cv2.addWeighted(overlay, 0.64, canvas, 0.36, 0.0, canvas)
    short_name = CAMERA_NAMES[spec.camera_id].replace("CAM ", "C").replace("  ", "_", 1)
    put_text(canvas, short_name, (spec.x + 12, spec.y + 20), 0.42, color, 2)
    put_text(canvas, "LIVE", (spec.x + spec.w - 58, spec.y + 20), 0.34, (210, 220, 220), 1)
    cv2.circle(canvas, (spec.x + spec.w - 14, spec.y + 15), 4, (45, 65, 255), -1, cv2.LINE_AA)

    footer_y = spec.y + spec.h + 21
    put_text(canvas, CAMERA_SUBTITLES[spec.camera_id], (spec.x + 10, footer_y), 0.35, (172, 184, 184), 1)
    put_text(canvas, f"{frame_idx:05d}", (spec.x + spec.w - 62, footer_y), 0.33, (118, 130, 130), 1)


def draw_footer(canvas: np.ndarray) -> None:
    x, y, w, h = 38, 874, 380, 140
    rounded_rect(canvas, (x, y), (x + w, y + h), (8, 12, 15), 8)
    rounded_rect(canvas, (x, y), (x + w, y + h), (82, 96, 100), 8, 1)
    legend = [
        ("COMMON ROUTE", "C01 -> C02 -> C03", ROUTE_COLORS["common"]),
        ("GOOD ROUTE", "C03 -> C04 -> C06", ROUTE_COLORS["good"]),
        ("DEFECT ROUTE", "C03 -> C05 -> C07", ROUTE_COLORS["defect"]),
    ]
    for idx, (name, route, color) in enumerate(legend):
        yy = y + 38 + idx * 38
        cv2.line(canvas, (x + 24, yy), (x + 74, yy), color, 5, cv2.LINE_AA)
        cv2.line(canvas, (x + 24, yy), (x + 74, yy), (245, 250, 255), 1, cv2.LINE_AA)
        put_text(canvas, name, (x + 94, yy + 5), 0.38, (230, 236, 236), 1)
        put_text(canvas, route, (x + 238, yy + 5), 0.36, (172, 184, 184), 1)


def compose_frame(
    frames: dict[str, np.ndarray],
    bbox_index: dict[tuple[str, int], list[dict[str, object]]],
    width: int,
    height: int,
    frame_idx: int,
    fps: float,
    metadata: dict[str, object],
    config: MonitorConfig,
    show_frame_id: bool,
) -> np.ndarray:
    canvas = draw_background(width, height)
    draw_header(canvas, frame_idx, fps, metadata, show_frame_id)
    draw_flow(canvas, frame_idx)
    for camera_id in CAMERA_IDS:
        draw_tile(
            canvas,
            LAYOUT_1080P[camera_id],
            frames[camera_id],
            frame_idx,
            bbox_index.get((camera_id, frame_idx), []),
            config,
        )
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
    config = MonitorConfig(show_bboxes=args.show_bboxes or args.bbox_mode == "on")
    annotations_path = Path(args.annotations_path) if args.annotations_path else default_annotations_path(input_dir)
    bbox_index = read_bbox_index(annotations_path) if config.show_bboxes else {}
    if config.show_bboxes:
        print(
            f"[monitor] Loaded bbox annotations: {annotations_path} "
            f"({sum(len(v) for v in bbox_index.values())} boxes; CARLA depth-visible annotations)"
        )

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
            canvas = compose_frame(
                frames,
                bbox_index,
                args.width,
                args.height,
                frame_idx,
                output_fps,
                metadata,
                config,
                args.show_frame_id,
            )
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
