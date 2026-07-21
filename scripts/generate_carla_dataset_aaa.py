#!/usr/bin/env python3
"""Generate an AAA-style Honda Smart Car Tracking CARLA dataset.

This variant intentionally keeps the same road, route, vehicles, parking logic,
and camera transforms as generate_carla_dataset.py. It only changes CARLA visual
quality settings and adds a cinematic CCTV post-process to the rendered MP4s.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import random
import shutil
import socket
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# The configs/ package lives at the repo root, one level above scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.presets import DatasetConfig, get_preset  # noqa: E402


_VIGNETTE_CACHE: dict[tuple[int, int], np.ndarray] = {}
_SCANLINE_CACHE: dict[tuple[int, int], np.ndarray] = {}

# ponytail: single-run script, so the active preset is a module global set once
# in main() rather than threaded through every CARLA helper. Read-only after that.
CONFIG: DatasetConfig = get_preset("aaa")


CAMERA_IDS = [
    "CAM_01_START",
    "CAM_02_TRANSIT",
    "CAM_03_JUNCTION_STATUS",
    "CAM_04_GOOD_ROUTE",
    "CAM_05_DEFECT_ROUTE",
    "CAM_06_GOOD_PARKING",
    "CAM_07_DEFECT_PARKING",
]


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    name: str
    role: str
    route_tags: tuple[str, ...]
    next_cameras: tuple[str, ...]
    map_xy: tuple[int, int]
    parking_slots: tuple[str, ...] = ()
    carla_location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    carla_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov: float = 70.0


@dataclass(frozen=True)
class CarSpec:
    tracking_id: str
    status: str
    route: tuple[str, ...]
    parking_slot_id: str
    color_bgr: tuple[int, int, int]
    start_offset_sec: float
    speed_factor: float


@dataclass(frozen=True)
class CarlaRoute:
    common: tuple[object, ...]
    good: tuple[object, ...]
    defect: tuple[object, ...]
    good_slots: dict[str, object]
    defect_slots: dict[str, object]


CAMERAS: tuple[CameraSpec, ...] = (
    CameraSpec(
        "CAM_01_START",
        "Start",
        "Vehicle detection at the production-line exit.",
        ("START",),
        ("CAM_02_TRANSIT",),
        (130, 240),
    ),
    CameraSpec(
        "CAM_02_TRANSIT",
        "Transit Route",
        "Cross-camera vehicle tracking from start to sorting junction.",
        ("TRANSIT",),
        ("CAM_03_JUNCTION_STATUS",),
        (330, 240),
    ),
    CameraSpec(
        "CAM_03_JUNCTION_STATUS",
        "Junction Status",
        "Status detection by turn direction: left is GOOD, right is DEFECT.",
        ("JUNCTION",),
        ("CAM_04_GOOD_ROUTE", "CAM_05_DEFECT_ROUTE"),
        (530, 240),
    ),
    CameraSpec(
        "CAM_04_GOOD_ROUTE",
        "Good Route",
        "Route segment for vehicles classified as GOOD.",
        ("GOOD",),
        ("CAM_06_GOOD_PARKING",),
        (720, 140),
    ),
    CameraSpec(
        "CAM_05_DEFECT_ROUTE",
        "Defect Route",
        "Route segment for vehicles classified as DEFECT.",
        ("DEFECT",),
        ("CAM_07_DEFECT_PARKING",),
        (720, 350),
    ),
    CameraSpec(
        "CAM_06_GOOD_PARKING",
        "Good Parking",
        "Parking slot detection for GOOD vehicles.",
        ("GOOD_PARKING",),
        (),
        (920, 140),
        ("G01", "G02", "G03", "G04", "G05", "G06"),
    ),
    CameraSpec(
        "CAM_07_DEFECT_PARKING",
        "Defect Parking",
        "Parking slot detection for DEFECT vehicles.",
        ("DEFECT_PARKING",),
        (),
        (920, 350),
        ("D01", "D02", "D03", "D04"),
    ),
)


SEGMENT_SECONDS = {
    "CAM_01_START": (0.0, 9.0),
    "CAM_02_TRANSIT": (7.0, 15.0),
    "CAM_03_JUNCTION_STATUS": (14.0, 23.0),
    "CAM_04_GOOD_ROUTE": (22.0, 31.0),
    "CAM_05_DEFECT_ROUTE": (22.0, 31.0),
    "CAM_06_GOOD_PARKING": (30.0, 42.0),
    "CAM_07_DEFECT_PARKING": (30.0, 42.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Honda Smart Car Tracking POC dataset with AAA-style CARLA/CCTV visuals."
    )
    # Flags that overlap a preset value default to None here and fall back to the
    # active preset in main(); pass one explicitly to override the preset.
    parser.add_argument(
        "--preset",
        default="aaa",
        help="Named dataset config from configs/presets.py. Default 'aaa' reproduces the shipped dataset.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--renderer", choices=("storyboard", "carla"), default="carla")
    parser.add_argument("--num-cars", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-map", default=None)
    parser.add_argument("--carla-timeout-sec", type=float, default=120.0)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--camera-ids",
        default="",
        help="Comma-separated camera IDs to render. Default renders all cameras.",
    )
    parser.add_argument(
        "--append-annotations",
        action="store_true",
        help="Append to annotations/bboxes.jsonl instead of replacing it. Useful for camera-by-camera resume.",
    )
    parser.add_argument(
        "--write-contact-sheets",
        action="store_true",
        help="Write sampled video contact sheets under the docs directory for quick camera QA.",
    )
    parser.add_argument(
        "--hide-camera-blockers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Temporarily hide nearby static map objects that block CAM_02/CAM_03 visibility.",
    )
    parser.add_argument(
        "--cctv-postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply AAA-style CCTV color, bloom, grain, vignette, scanline, and REC timestamp styling.",
    )
    parser.add_argument(
        "--cctv-grain-strength",
        type=float,
        default=None,
        help="Per-pixel grain standard deviation for CCTV post-process. Use 0 to disable grain.",
    )
    parser.add_argument(
        "--temporal-stabilization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Blend nearby frames slightly to reduce CARLA capture flicker and make output FPS feel steadier.",
    )
    parser.add_argument(
        "--temporal-blend-strength",
        type=float,
        default=None,
        help="Frame-to-frame blend strength for temporal stabilization.",
    )
    parser.add_argument(
        "--wheel-motion-hints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send per-frame velocity/angular-velocity hints to vehicles for better motion rendering.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove previous generated outputs.")
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help=(
            "Re-project bbox annotations only against the running CARLA server: reuse the existing "
            "videos, skip RGB capture/encode and CCTV post-process, and ignore the per-camera segment "
            "time windows so each vehicle is annotated for its full in-frame duration."
        ),
    )
    parser.add_argument(
        "--min-bbox-px",
        type=int,
        default=4,
        help="Minimum projected bbox width/height (pixels) kept in --annotations-only mode.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 5 <= args.num_cars <= 6:
        raise ValueError("--num-cars must be between 5 and 6 for the focused POC dataset.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.duration_sec < 45:
        raise ValueError("--duration-sec must be at least 45 seconds.")
    if args.width < 640 or args.height < 360:
        raise ValueError("--width/--height are too small for vehicle tracking review.")
    if args.carla_timeout_sec <= 0:
        raise ValueError("--carla-timeout-sec must be positive.")
    if args.cctv_grain_strength < 0:
        raise ValueError("--cctv-grain-strength must be non-negative.")
    if not 0.0 <= args.temporal_blend_strength <= 0.35:
        raise ValueError("--temporal-blend-strength must be between 0.0 and 0.35.")
    if args.camera_ids:
        valid_camera_ids = set(CAMERA_IDS)
        requested = {camera_id.strip() for camera_id in args.camera_ids.split(",") if camera_id.strip()}
        unknown = sorted(requested - valid_camera_ids)
        if unknown:
            raise ValueError(f"Unknown --camera-ids values: {', '.join(unknown)}")


def selected_cameras(camera_ids: str) -> list[CameraSpec]:
    if not camera_ids:
        return list(CAMERAS)
    requested = {camera_id.strip() for camera_id in camera_ids.split(",") if camera_id.strip()}
    return [camera for camera in CAMERAS if camera.camera_id in requested]


def log_step(message: str) -> None:
    print(f"[generate] {message}", flush=True)


class ProgressBar:
    def __init__(self, label: str, total: int, width: int = 32) -> None:
        self.label = label
        self.total = max(1, total)
        self.width = width
        self.current = 0
        self.started_at = time.monotonic()
        self.last_render_at = 0.0

    def update(self, current: int | None = None, advance: int = 1, extra: str = "") -> None:
        if current is None:
            self.current += advance
        else:
            self.current = current
        self.current = max(0, min(self.current, self.total))
        now = time.monotonic()
        if self.current < self.total and now - self.last_render_at < 0.2:
            return
        self.last_render_at = now
        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.started_at
        rate = self.current / elapsed if elapsed > 0 else 0.0
        suffix = f" | {extra}" if extra else ""
        print(
            f"\r[{bar}] {self.label} {self.current}/{self.total} "
            f"({ratio * 100:5.1f}%) {rate:5.1f}/s{suffix}",
            end="",
            flush=True,
        )
        if self.current >= self.total:
            print(flush=True)


def make_cars(num_cars: int, random_seed: int) -> list[CarSpec]:
    colors = list(CONFIG.car_colors_bgr)
    cars: list[CarSpec] = []
    rng = random.Random(random_seed)
    good_slots = ["G01", "G02", "G03", "G04", "G05", "G06"]
    defect_slots = ["D01", "D02", "D03", "D04"]
    start_offset_sec = 0.0
    good_count = 0
    defect_count = 0

    for idx in range(num_cars):
        status = "DEFECT" if idx % 2 == 1 else "GOOD"
        if status == "GOOD":
            route = (
                "CAM_01_START",
                "CAM_02_TRANSIT",
                "CAM_03_JUNCTION_STATUS",
                "CAM_04_GOOD_ROUTE",
                "CAM_06_GOOD_PARKING",
            )
            slot = good_slots[good_count % len(good_slots)]
            good_count += 1
        else:
            route = (
                "CAM_01_START",
                "CAM_02_TRANSIT",
                "CAM_03_JUNCTION_STATUS",
                "CAM_05_DEFECT_ROUTE",
                "CAM_07_DEFECT_PARKING",
            )
            slot = defect_slots[defect_count % len(defect_slots)]
            defect_count += 1

        cars.append(
            CarSpec(
                tracking_id=f"TRK_{idx + 1:04d}",
                status=status,
                route=route,
                parking_slot_id=slot,
                color_bgr=colors[idx % len(colors)],
                start_offset_sec=start_offset_sec,
                speed_factor=CONFIG.speed_factor_base
                + (idx % CONFIG.speed_factor_cycle) * CONFIG.speed_factor_step,
            )
        )
        start_offset_sec += rng.uniform(*CONFIG.start_offset_range_sec)
    return cars


def ensure_dirs(output_dir: Path, docs_dir: Path, clean: bool) -> dict[str, Path]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    paths = {
        "videos": output_dir / "videos",
        "metadata": output_dir / "metadata",
        "annotations": output_dir / "annotations",
        "docs": docs_dir,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_camera_graph(path: Path) -> None:
    graph = {
        "dataset": "carla_honda_poc_aaa",
        "renderer": "storyboard",
        "carla_map": "Town05_Opt/Town05",
        "note": "Storyboard preview only. Final POC dataset must be generated with --renderer carla.",
        "status_rule": {
            "camera_id": "CAM_03_JUNCTION_STATUS",
            "left_turn": "GOOD",
            "right_turn": "DEFECT",
        },
        "cameras": [asdict(camera) for camera in CAMERAS],
    }
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")


def write_cars_csv(path: Path, cars: Iterable[CarSpec]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "tracking_id",
                "status",
                "route",
                "parking_slot_id",
                "start_offset_sec",
                "speed_factor",
            ],
        )
        writer.writeheader()
        for car in cars:
            writer.writerow(
                {
                    "tracking_id": car.tracking_id,
                    "status": car.status,
                    "route": ">".join(car.route),
                    "parking_slot_id": car.parking_slot_id,
                    "start_offset_sec": f"{car.start_offset_sec:.2f}",
                    "speed_factor": f"{car.speed_factor:.2f}",
                }
            )


def segment_window(car: CarSpec, camera_id: str) -> tuple[float, float] | None:
    if camera_id not in car.route:
        return None
    base_start, base_end = SEGMENT_SECONDS[camera_id]
    return (
        car.start_offset_sec + base_start / car.speed_factor,
        car.start_offset_sec + base_end / car.speed_factor,
    )


def position_for(camera_id: str, car: CarSpec, progress: float, width: int, height: int) -> tuple[int, int]:
    lane_offset = (int(car.tracking_id[-2:]) % 4 - 1.5) * 34
    if camera_id == "CAM_03_JUNCTION_STATUS":
        x0, y0 = int(width * 0.15), int(height * 0.58) + int(lane_offset * 0.4)
        x1 = int(width * 0.82)
        y1 = int(height * 0.28 if car.status == "GOOD" else height * 0.78)
        bend = math.sin(progress * math.pi) * (80 if car.status == "GOOD" else -40)
        x = int(x0 + (x1 - x0) * progress)
        y = int(y0 + (y1 - y0) * progress - bend)
        return x, y
    if camera_id.endswith("PARKING"):
        slot_idx = int(car.parking_slot_id[1:]) - 1
        slots = 6 if car.status == "GOOD" else 4
        slot_w = width / (slots + 1)
        target_x = int(slot_w * (slot_idx + 1))
        x = int((width * 0.08) * (1 - progress) + target_x * progress)
        y = int(height * (0.62 + 0.08 * (slot_idx % 2)))
        return x, y
    x = int(width * (-0.10 + 1.20 * progress))
    y = int(height * 0.58 + lane_offset)
    return x, y


def bbox_for(cx: int, cy: int, progress: float, width: int, height: int) -> list[int]:
    scale = 0.85 + 0.35 * math.sin(progress * math.pi)
    box_w = int(width * 0.105 * scale)
    box_h = int(height * 0.070 * scale)
    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(width - 1, cx + box_w // 2)
    y2 = min(height - 1, cy + box_h // 2)
    return [x1, y1, x2, y2]


def visible_bbox(
    camera_id: str,
    car: CarSpec,
    timestamp_sec: float,
    width: int,
    height: int,
) -> list[int] | None:
    window = segment_window(car, camera_id)
    if window is None:
        return None
    start, end = window
    if timestamp_sec < start or timestamp_sec > end:
        return None
    progress = (timestamp_sec - start) / max(0.001, end - start)
    cx, cy = position_for(camera_id, car, progress, width, height)
    bbox = bbox_for(cx, cy, progress, width, height)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    if bbox[2] - bbox[0] < 36 or bbox[3] - bbox[1] < 24:
        return None
    return bbox


def draw_base_scene(draw: ImageDraw.ImageDraw, camera: CameraSpec, width: int, height: int) -> None:
    draw.rectangle((0, 0, width, height), fill=(42, 48, 54))
    draw.rectangle((0, int(height * 0.45), width, int(height * 0.75)), fill=(68, 72, 76))
    for y in (int(height * 0.52), int(height * 0.65)):
        for x in range(0, width, 90):
            draw.rectangle((x, y, x + 45, y + 5), fill=(214, 205, 120))

    if camera.camera_id == "CAM_03_JUNCTION_STATUS":
        draw.polygon(
            [
                (int(width * 0.45), int(height * 0.48)),
                (width, int(height * 0.16)),
                (width, int(height * 0.32)),
                (int(width * 0.50), int(height * 0.62)),
            ],
            fill=(70, 74, 78),
        )
        draw.polygon(
            [
                (int(width * 0.45), int(height * 0.62)),
                (width, int(height * 0.83)),
                (width, int(height * 0.96)),
                (int(width * 0.50), int(height * 0.68)),
            ],
            fill=(70, 74, 78),
        )
        draw.text((32, 32), "LEFT = GOOD / RIGHT = DEFECT", fill=(240, 240, 240))

    if camera.parking_slots:
        slot_count = len(camera.parking_slots)
        slot_w = width / (slot_count + 1)
        for idx, slot in enumerate(camera.parking_slots):
            cx = int(slot_w * (idx + 1))
            x1, x2 = cx - 58, cx + 58
            y1, y2 = int(height * 0.50), int(height * 0.88)
            draw.rectangle((x1, y1, x2, y2), outline=(235, 235, 235), width=3)
            draw.text((x1 + 24, y1 + 10), slot, fill=(255, 255, 255))

    draw.rectangle((18, height - 62, 395, height - 18), fill=(18, 24, 30))
    draw.text((32, height - 51), f"{camera.camera_id} | {camera.name}", fill=(245, 245, 245))


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_car(
    draw: ImageDraw.ImageDraw,
    camera_id: str,
    car: CarSpec,
    bbox: list[int],
    width: int,
    height: int,
) -> None:
    x1, y1, x2, y2 = bbox
    color = tuple(reversed(car.color_bgr))
    draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=color, outline=(25, 30, 35), width=3)
    wind_h = max(10, int((y2 - y1) * 0.35))
    draw.rectangle((x1 + 12, y1 + 8, x2 - 12, y1 + 8 + wind_h), fill=(55, 75, 90))
    draw.ellipse((x1 + 8, y2 - 13, x1 + 24, y2 + 3), fill=(20, 20, 20))
    draw.ellipse((x2 - 24, y2 - 13, x2 - 8, y2 + 3), fill=(20, 20, 20))

    label = f"{car.tracking_id} {car.status}"
    draw.text((x1, max(4, y1 - 22)), label, fill=(255, 255, 255))


def write_events(path: Path, cars: Iterable[CarSpec]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for car in cars:
            for camera_id in car.route:
                start, end = segment_window(car, camera_id) or (0.0, 0.0)
                event = {
                    "tracking_id": car.tracking_id,
                    "status": car.status,
                    "camera_id": camera_id,
                    "enter_timestamp_sec": round(start, 3),
                    "exit_timestamp_sec": round(end, 3),
                    "parking_slot_id": car.parking_slot_id if camera_id.endswith("PARKING") else "",
                }
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def render_storyboard(
    paths: dict[str, Path],
    cars: list[CarSpec],
    fps: int,
    duration_sec: float,
    width: int,
    height: int,
) -> None:
    frame_count = int(duration_sec * fps)
    bbox_path = paths["annotations"] / "bboxes.jsonl"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with bbox_path.open("w", encoding="utf-8") as bbox_fh:
        for camera in CAMERAS:
            progress = ProgressBar(f"storyboard {camera.camera_id}", frame_count)
            video_path = paths["videos"] / f"{camera.camera_id}.mp4"
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {video_path}")

            for frame_id in range(frame_count):
                timestamp_sec = frame_id / fps
                img = Image.new("RGB", (width, height))
                draw = ImageDraw.Draw(img)
                draw_base_scene(draw, camera, width, height)

                visible: list[tuple[CarSpec, list[int]]] = []
                for car in cars:
                    bbox = visible_bbox(camera.camera_id, car, timestamp_sec, width, height)
                    if bbox:
                        visible.append((car, bbox))
                visible.sort(key=lambda item: item[1][1])

                for car, bbox in visible:
                    draw_car(draw, camera.camera_id, car, bbox, width, height)
                    record = {
                        "tracking_id": car.tracking_id,
                        "status": car.status,
                        "route": list(car.route),
                        "camera_id": camera.camera_id,
                        "frame_id": frame_id,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "bbox": bbox,
                        "bbox_source": "storyboard_2d",
                        "parking_slot_id": car.parking_slot_id if camera.camera_id.endswith("PARKING") else "",
                    }
                    bbox_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
                progress.update(frame_id + 1)
            writer.release()


def import_carla_module():
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is not installed. Add the CARLA 0.9.15 PythonAPI egg "
            "to PYTHONPATH or install the matching API package on the CARLA machine."
        ) from exc
    return carla


def transform_to_dict(transform: object) -> dict[str, dict[str, float]]:
    return {
        "location": {
            "x": round(transform.location.x, 4),
            "y": round(transform.location.y, 4),
            "z": round(transform.location.z, 4),
        },
        "rotation": {
            "pitch": round(transform.rotation.pitch, 4),
            "yaw": round(transform.rotation.yaw, 4),
            "roll": round(transform.rotation.roll, 4),
        },
    }


def vector_length(x: float, y: float, z: float = 0.0) -> float:
    return math.sqrt(x * x + y * y + z * z)


def yaw_to_forward(carla, yaw_deg: float):
    yaw = math.radians(yaw_deg)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


def yaw_to_right(carla, yaw_deg: float):
    yaw = math.radians(yaw_deg + 90.0)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


def look_at_rotation(carla, origin: object, target: object):
    dx = target.x - origin.x
    dy = target.y - origin.y
    dz = target.z - origin.z
    distance_xy = max(0.001, vector_length(dx, dy))
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, distance_xy))
    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def lerp_location(carla, start: object, end: object, alpha: float):
    return carla.Location(
        x=start.x + (end.x - start.x) * alpha,
        y=start.y + (end.y - start.y) * alpha,
        z=start.z + (end.z - start.z) * alpha,
    )


def make_transform_between(carla, start: object, end: object, alpha: float):
    alpha = max(0.0, min(1.0, alpha))
    loc = lerp_location(carla, start, end, alpha)
    yaw = math.degrees(math.atan2(end.y - start.y, end.x - start.x))
    loc.z += 0.10
    return carla.Transform(loc, carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))


def project_to_road(carla, world_map: object, location: object):
    waypoint = world_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    transform = waypoint.transform
    transform.location.z += 0.05
    return transform


def sample_forward_route(carla, world_map: object, spawn_transform: object, count: int, step_m: float) -> list[object]:
    waypoint = world_map.get_waypoint(
        spawn_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    transforms = [waypoint.transform]
    for _ in range(count - 1):
        next_waypoints = waypoint.next(step_m)
        if not next_waypoints:
            break
        waypoint = next_waypoints[0]
        transforms.append(waypoint.transform)
    if len(transforms) < count:
        forward = yaw_to_forward(carla, transforms[-1].rotation.yaw)
        while len(transforms) < count:
            prev = transforms[-1]
            loc = carla.Location(
                x=prev.location.x + forward.x * step_m,
                y=prev.location.y + forward.y * step_m,
                z=prev.location.z,
            )
            transforms.append(project_to_road(carla, world_map, loc))
    return transforms


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def interpolate_yaw(start_yaw: float, end_yaw: float, alpha: float) -> float:
    return start_yaw + normalize_angle(end_yaw - start_yaw) * max(0.0, min(1.0, alpha))


def extend_waypoint_route(
    start_waypoint: object,
    count: int,
    step_m: float,
    preferred_yaw: float | None = None,
) -> list[object]:
    waypoint = start_waypoint
    transforms = [waypoint.transform]
    for _ in range(count - 1):
        next_waypoints = waypoint.next(step_m)
        if not next_waypoints:
            break
        if preferred_yaw is None:
            waypoint = min(
                next_waypoints,
                key=lambda candidate: abs(
                    normalize_angle(candidate.transform.rotation.yaw - waypoint.transform.rotation.yaw)
                ),
            )
        else:
            waypoint = min(
                next_waypoints,
                key=lambda candidate: abs(normalize_angle(candidate.transform.rotation.yaw - preferred_yaw)),
            )
        preferred_yaw = waypoint.transform.rotation.yaw
        transforms.append(waypoint.transform)
    return transforms


def find_route_through_junction(carla, world_map: object, spawn_transform: object) -> CarlaRoute | None:
    waypoint = world_map.get_waypoint(
        spawn_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    common_waypoints = [waypoint]
    step_m = 4.0

    for _ in range(36):
        next_waypoints = waypoint.next(step_m)
        if not next_waypoints:
            return None
        if len(next_waypoints) >= 2 and len(common_waypoints) >= 8:
            current_yaw = waypoint.transform.rotation.yaw
            sorted_next = sorted(
                next_waypoints,
                key=lambda candidate: normalize_angle(candidate.transform.rotation.yaw - current_yaw),
            )
            left_wp = sorted_next[0]
            right_wp = sorted_next[-1]
            if left_wp.id == right_wp.id:
                return None
            common = [wp.transform for wp in common_waypoints]
            good = extend_waypoint_route(left_wp, count=18, step_m=step_m)
            defect = extend_waypoint_route(right_wp, count=18, step_m=step_m)
            if len(good) < 10 or len(defect) < 10:
                return None
            return CarlaRoute(
                common=tuple(common),
                good=tuple(good),
                defect=tuple(defect),
                good_slots=make_parking_slots(carla, good, "G", 6, side_sign=-1.0),
                defect_slots=make_parking_slots(carla, defect, "D", 4, side_sign=1.0),
            )

        waypoint = min(
            next_waypoints,
            key=lambda candidate: abs(
                normalize_angle(candidate.transform.rotation.yaw - waypoint.transform.rotation.yaw)
            ),
        )
        common_waypoints.append(waypoint)
    return None


def make_parking_slots(
    carla,
    branch_transforms: list[object],
    prefix: str,
    count: int,
    side_sign: float,
) -> dict[str, object]:
    base = branch_transforms[-1]
    forward = yaw_to_forward(carla, base.rotation.yaw)
    right = yaw_to_right(carla, base.rotation.yaw)
    result = {}
    for idx in range(count):
        row = idx // 3
        col = idx % 3
        loc = carla.Location(
            x=base.location.x + forward.x * (col * 4.2) + right.x * side_sign * (6.5 + row * 4.5),
            y=base.location.y + forward.y * (col * 4.2) + right.y * side_sign * (6.5 + row * 4.5),
            z=base.location.z + 0.10,
        )
        result[f"{prefix}{idx + 1:02d}"] = carla.Transform(
            loc,
            carla.Rotation(pitch=0.0, yaw=base.rotation.yaw + side_sign * 90.0, roll=0.0),
        )
    return result


def build_carla_route(carla, world: object) -> CarlaRoute:
    world_map = world.get_map()
    spawn_points = sorted(
        world_map.get_spawn_points(),
        key=lambda transform: (round(transform.location.x, 2), round(transform.location.y, 2)),
    )
    if not spawn_points:
        raise RuntimeError("CARLA map returned no spawn points.")

    for spawn_transform in spawn_points:
        route = find_route_through_junction(carla, world_map, spawn_transform)
        if route is not None:
            validate_carla_route(route)
            return route
    raise RuntimeError("Could not find a usable start -> junction -> two-branch route in the selected map.")


def parking_maneuver_transforms(carla, road_transform: object, slot_transform: object) -> tuple[object, ...]:
    road_yaw = road_transform.rotation.yaw
    slot_yaw = slot_transform.rotation.yaw
    slot_forward = yaw_to_forward(carla, slot_yaw)
    road_forward = yaw_to_forward(carla, road_yaw)

    staging = carla.Transform(
        carla.Location(
            x=slot_transform.location.x - slot_forward.x * 7.5,
            y=slot_transform.location.y - slot_forward.y * 7.5,
            z=slot_transform.location.z,
        ),
        carla.Rotation(pitch=0.0, yaw=road_yaw, roll=0.0),
    )
    overshoot = carla.Transform(
        carla.Location(
            x=slot_transform.location.x + road_forward.x * 5.5 - slot_forward.x * 2.5,
            y=slot_transform.location.y + road_forward.y * 5.5 - slot_forward.y * 2.5,
            z=slot_transform.location.z,
        ),
        carla.Rotation(pitch=0.0, yaw=road_yaw, roll=0.0),
    )
    reverse_entry = carla.Transform(
        carla.Location(
            x=slot_transform.location.x - slot_forward.x * 3.6,
            y=slot_transform.location.y - slot_forward.y * 3.6,
            z=slot_transform.location.z,
        ),
        carla.Rotation(pitch=0.0, yaw=interpolate_yaw(road_yaw, slot_yaw, 0.65), roll=0.0),
    )
    correction = carla.Transform(
        carla.Location(
            x=slot_transform.location.x + slot_forward.x * 0.7,
            y=slot_transform.location.y + slot_forward.y * 0.7,
            z=slot_transform.location.z,
        ),
        carla.Rotation(pitch=0.0, yaw=slot_yaw + 3.0, roll=0.0),
    )
    return (staging, overshoot, reverse_entry, slot_transform, correction, slot_transform)


def path_for_car(carla, route: CarlaRoute, car: CarSpec) -> tuple[object, ...]:
    if car.status == "GOOD":
        slot = route.good_slots[car.parking_slot_id]
        return tuple(route.common) + route.good + parking_maneuver_transforms(carla, route.good[-1], slot)
    slot = route.defect_slots[car.parking_slot_id]
    return tuple(route.common) + route.defect + parking_maneuver_transforms(carla, route.defect[-1], slot)


def validate_carla_route(route: CarlaRoute) -> None:
    if len(route.common) < 8:
        raise RuntimeError("Route common segment is too short for start/transit/junction cameras.")
    if len(route.good) < 10 or len(route.defect) < 10:
        raise RuntimeError("Route branches are too short for route and parking cameras.")
    expected_good = {f"G{idx + 1:02d}" for idx in range(6)}
    expected_defect = {f"D{idx + 1:02d}" for idx in range(4)}
    if set(route.good_slots) != expected_good:
        raise RuntimeError("GOOD parking slots are incomplete.")
    if set(route.defect_slots) != expected_defect:
        raise RuntimeError("DEFECT parking slots are incomplete.")
    for camera_id, next_camera_ids in {
        "CAM_01_START": ("CAM_02_TRANSIT",),
        "CAM_02_TRANSIT": ("CAM_03_JUNCTION_STATUS",),
        "CAM_03_JUNCTION_STATUS": ("CAM_04_GOOD_ROUTE", "CAM_05_DEFECT_ROUTE"),
        "CAM_04_GOOD_ROUTE": ("CAM_06_GOOD_PARKING",),
        "CAM_05_DEFECT_ROUTE": ("CAM_07_DEFECT_PARKING",),
    }.items():
        end = SEGMENT_SECONDS[camera_id][1]
        for next_camera_id in next_camera_ids:
            start = SEGMENT_SECONDS[next_camera_id][0]
            if end < start:
                raise RuntimeError(f"Camera segment {camera_id} has no overlap with {next_camera_id}.")


def route_transform_at(carla, transforms: tuple[object, ...], progress: float):
    progress = max(0.0, min(1.0, progress))
    if len(transforms) == 1:
        return transforms[0]
    scaled = progress * (len(transforms) - 1)
    idx = min(len(transforms) - 2, int(scaled))
    alpha = scaled - idx
    start = transforms[idx]
    end = transforms[idx + 1]
    loc = lerp_location(carla, start.location, end.location, alpha)
    loc.z += 0.10
    yaw = interpolate_yaw(start.rotation.yaw, end.rotation.yaw, alpha)
    return carla.Transform(loc, carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))


def clone_transform_with_offset(carla, transform: object, lateral_m: float = 0.0, z_m: float = 0.0):
    right = yaw_to_right(carla, transform.rotation.yaw)
    location = carla.Location(
        x=transform.location.x + right.x * lateral_m,
        y=transform.location.y + right.y * lateral_m,
        z=transform.location.z + z_m,
    )
    rotation = carla.Rotation(
        pitch=transform.rotation.pitch,
        yaw=transform.rotation.yaw,
        roll=transform.rotation.roll,
    )
    return carla.Transform(location, rotation)


def hidden_vehicle_transform(carla, idx: int):
    return carla.Transform(
        carla.Location(x=-10000.0, y=-10000.0 - idx * 12.0, z=-100.0),
        carla.Rotation(),
    )


def set_actor_motion_hint(carla, actor: object, previous_transform: object, current_transform: object, delta_sec: float) -> None:
    if delta_sec <= 0.0:
        return
    try:
        velocity = carla.Vector3D(
            x=(current_transform.location.x - previous_transform.location.x) / delta_sec,
            y=(current_transform.location.y - previous_transform.location.y) / delta_sec,
            z=(current_transform.location.z - previous_transform.location.z) / delta_sec,
        )
        yaw_delta = normalize_angle(current_transform.rotation.yaw - previous_transform.rotation.yaw)
        angular_velocity = carla.Vector3D(x=0.0, y=0.0, z=yaw_delta / delta_sec)
        actor.set_target_velocity(velocity)
        actor.set_target_angular_velocity(angular_velocity)
    except (AttributeError, RuntimeError):
        pass


def clear_actor_motion_hint(carla, actor: object) -> None:
    try:
        actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    except (AttributeError, RuntimeError):
        pass


def set_vehicle_transforms_for_time(
    carla,
    cars: list[CarSpec],
    vehicle_actors: dict[str, object],
    car_paths: dict[str, tuple[object, ...]],
    timestamp_sec: float,
    duration_sec: float,
    delta_sec: float,
    wheel_motion_hints: bool,
) -> None:
    for car_idx, car in enumerate(cars):
        start = car.start_offset_sec
        route_base_duration = max(end for _, end in SEGMENT_SECONDS.values())
        drive_duration = route_base_duration / car.speed_factor + 4.0
        end = min(duration_sec - 2.0, start + drive_duration)
        end = max(start + 1.0, end)
        progress = (timestamp_sec - start) / (end - start)
        actor = vehicle_actors[car.tracking_id]
        if timestamp_sec < start:
            actor.set_transform(hidden_vehicle_transform(carla, car_idx))
            if wheel_motion_hints:
                clear_actor_motion_hint(carla, actor)
        else:
            path = car_paths[car.tracking_id]
            current_transform = route_transform_at(carla, path, progress)
            actor.set_transform(current_transform)
            if wheel_motion_hints:
                previous_time = max(start, timestamp_sec - delta_sec)
                previous_progress = (previous_time - start) / (end - start)
                previous_transform = route_transform_at(carla, path, previous_progress)
                set_actor_motion_hint(carla, actor, previous_transform, current_transform, delta_sec)


def camera_targets_from_route(route: CarlaRoute) -> dict[str, object]:
    common_last = len(route.common) - 1
    good_last = len(route.good) - 1
    defect_last = len(route.defect) - 1
    return {
        "CAM_01_START": route.common[min(2, common_last)].location,
        "CAM_02_TRANSIT": route.common[min(8, common_last)].location,
        "CAM_03_JUNCTION_STATUS": route.common[common_last].location,
        "CAM_04_GOOD_ROUTE": route.good[min(10, good_last)].location,
        "CAM_05_DEFECT_ROUTE": route.defect[min(10, defect_last)].location,
        "CAM_06_GOOD_PARKING": average_transform_location(route.good_slots.values()),
        "CAM_07_DEFECT_PARKING": average_transform_location(route.defect_slots.values()),
    }


def average_transform_location(transforms: Iterable[object]) -> object:
    transforms = tuple(transforms)
    first = transforms[0]
    location = type(first.location)()
    location.x = sum(transform.location.x for transform in transforms) / len(transforms)
    location.y = sum(transform.location.y for transform in transforms) / len(transforms)
    location.z = sum(transform.location.z for transform in transforms) / len(transforms)
    return location


def average_locations(carla, locations: Iterable[object]) -> object:
    locations = tuple(locations)
    return carla.Location(
        x=sum(location.x for location in locations) / len(locations),
        y=sum(location.y for location in locations) / len(locations),
        z=sum(location.z for location in locations) / len(locations),
    )


def build_camera_transforms(carla, route: CarlaRoute) -> dict[str, object]:
    targets = camera_targets_from_route(route)
    look_targets = dict(targets)
    common_last = len(route.common) - 1
    good_last = len(route.good) - 1
    defect_last = len(route.defect) - 1

    targets["CAM_02_TRANSIT"] = route.common[min(12, common_last)].location
    look_targets["CAM_02_TRANSIT"] = route.common[min(16, common_last)].location
    targets["CAM_03_JUNCTION_STATUS"] = route.common[common_last].location
    look_targets["CAM_03_JUNCTION_STATUS"] = average_locations(
        carla,
        (
            route.common[max(common_last - 2, 0)].location,
            route.good[min(2, good_last)].location,
            route.defect[min(5, defect_last)].location,
        ),
    )
    look_targets["CAM_05_DEFECT_ROUTE"] = route.defect[min(13, defect_last)].location

    yaw_by_camera = {
        "CAM_01_START": route.common[min(3, len(route.common) - 1)].rotation.yaw,
        "CAM_02_TRANSIT": route.common[min(12, len(route.common) - 1)].rotation.yaw,
        "CAM_03_JUNCTION_STATUS": route.common[-1].rotation.yaw,
        "CAM_04_GOOD_ROUTE": route.good[min(10, len(route.good) - 1)].rotation.yaw,
        "CAM_05_DEFECT_ROUTE": route.defect[min(10, len(route.defect) - 1)].rotation.yaw,
        "CAM_06_GOOD_PARKING": route.good[-1].rotation.yaw,
        "CAM_07_DEFECT_PARKING": route.defect[-1].rotation.yaw,
    }
    camera_plan = CONFIG.camera_plan
    transforms: dict[str, object] = {}
    for camera in CAMERAS:
        target = targets[camera.camera_id]
        plan = camera_plan[camera.camera_id]
        offset = yaw_to_forward(carla, yaw_by_camera[camera.camera_id] + plan["yaw_offset"])
        location = carla.Location(
            x=target.x + offset.x * plan["distance"],
            y=target.y + offset.y * plan["distance"],
            z=target.z + plan["height"],
        )
        transforms[camera.camera_id] = carla.Transform(
            location,
            look_at_rotation(carla, location, look_targets[camera.camera_id]),
        )
    return transforms


def carla_camera_fovs() -> dict[str, float]:
    return {camera.camera_id: CONFIG.camera_fovs[camera.camera_id] for camera in CAMERAS}


def build_projection_matrix(width: int, height: int, fov: float) -> np.ndarray:
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    matrix = np.identity(3)
    matrix[0, 0] = matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return matrix


def get_image_point(location: object, intrinsic: np.ndarray, world_to_camera: np.ndarray) -> tuple[float, float, float]:
    point = np.array([location.x, location.y, location.z, 1.0])
    point_camera = np.dot(world_to_camera, point)
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])
    if point_camera[2] <= 0.0:
        return 0.0, 0.0, float(point_camera[2])
    point_img = np.dot(intrinsic, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return float(point_img[0]), float(point_img[1]), float(point_camera[2])


def project_actor_bbox_points(
    actor: object,
    camera_transform: object,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[list[int] | None, list[tuple[float, float, float]]]:
    world_to_camera = np.array(camera_transform.get_inverse_matrix())
    vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
    projected = [get_image_point(vertex, intrinsic, world_to_camera) for vertex in vertices]
    visible = [(x, y) for x, y, depth in projected if depth > 0.0]
    if len(visible) < 4:
        return None, projected
    xs = [point[0] for point in visible]
    ys = [point[1] for point in visible]
    x1 = max(0, int(math.floor(min(xs))))
    y1 = max(0, int(math.floor(min(ys))))
    x2 = min(width - 1, int(math.ceil(max(xs))))
    y2 = min(height - 1, int(math.ceil(max(ys))))
    if x2 <= x1 or y2 <= y1:
        return None, projected
    if x2 < 0 or y2 < 0 or x1 >= width or y1 >= height:
        return None, projected
    return [x1, y1, x2, y2], projected


def project_actor_bbox(
    actor: object,
    camera_transform: object,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> list[int] | None:
    bbox, _projected = project_actor_bbox_points(actor, camera_transform, intrinsic, width, height)
    return bbox


def depth_image_to_meters(image: object) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4)).astype(np.float32)
    # CARLA depth cameras encode normalized depth in RGB; raw_data arrives as BGRA.
    normalized = (array[:, :, 2] + array[:, :, 1] * 256.0 + array[:, :, 0] * 65536.0) / (256.0**3 - 1.0)
    return normalized * 1000.0


def semantic_image_to_tags(image: object) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    # CARLA semantic raw frames encode the class tag in a color channel. Taking the
    # channel max keeps this robust across BGRA/RGBA channel-order differences.
    return np.maximum.reduce((array[:, :, 0], array[:, :, 1], array[:, :, 2]))


def visible_depth_at(depth_meters: np.ndarray, x: float, y: float, radius: int = 2) -> float | None:
    height, width = depth_meters.shape[:2]
    px = int(round(x))
    py = int(round(y))
    if px < 0 or px >= width or py < 0 or py >= height:
        return None
    x1 = max(0, px - radius)
    x2 = min(width, px + radius + 1)
    y1 = max(0, py - radius)
    y2 = min(height, py + radius + 1)
    crop = depth_meters[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    finite = crop[np.isfinite(crop)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, 20))


def bbox_has_depth_evidence(
    depth_meters: np.ndarray,
    bbox: list[int],
    actor_depth: float,
    tolerance_m: float,
) -> float:
    x1, y1, x2, y2 = bbox
    crop = depth_meters[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return 0.0
    stride_y = max(1, crop.shape[0] // 48)
    stride_x = max(1, crop.shape[1] // 48)
    sample = crop[::stride_y, ::stride_x]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return 0.0
    close = np.abs(finite - actor_depth) <= tolerance_m
    return float(np.count_nonzero(close) / finite.size)


def bbox_has_vehicle_semantic_evidence(semantic_tags: np.ndarray, bbox: list[int]) -> tuple[bool, int, float]:
    x1, y1, x2, y2 = bbox
    crop = semantic_tags[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return False, 0, 0.0
    vehicle_mask = crop == 10  # carla.CityObjectLabel.Vehicles
    vehicle_pixels = int(np.count_nonzero(vehicle_mask))
    ratio = float(vehicle_pixels / crop.size)
    min_pixels = max(8, min(80, int(crop.size * 0.002)))
    return vehicle_pixels >= min_pixels, vehicle_pixels, ratio


def yolo_like_visible_actor_bbox(
    actor: object,
    camera_transform: object,
    intrinsic: np.ndarray,
    depth_meters: np.ndarray,
    semantic_tags: np.ndarray | None,
    width: int,
    height: int,
    min_bbox_px: int = 10,
    relaxed: bool = False,
) -> tuple[list[int] | None, dict[str, object]]:
    bbox, projected = project_actor_bbox_points(actor, camera_transform, intrinsic, width, height)
    if bbox is None:
        return None, {"visibility_reject": "projection_outside_view"}

    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    area = box_w * box_h
    # Relaxed mode keeps far/small but genuinely visible vehicles (e.g. junction,
    # defect route, parking) instead of culling them like the strict default.
    min_w = min_bbox_px if relaxed else 10
    min_h = min_bbox_px if relaxed else 8
    min_area = (min_bbox_px * min_bbox_px) if relaxed else 160
    if box_w < min_w or box_h < min_h or area < min_area:
        return None, {"visibility_reject": "bbox_too_small", "projected_bbox": bbox}

    projected_in_frame = [
        (x, y, depth)
        for x, y, depth in projected
        if depth > 0.0 and 0 <= x < width and 0 <= y < height
    ]
    if not projected_in_frame:
        return None, {"visibility_reject": "no_projected_points_in_frame", "projected_bbox": bbox}

    actor_depth = float(np.median([depth for _x, _y, depth in projected_in_frame]))
    tolerance_m = max(3.0, actor_depth * 0.12) if relaxed else max(1.5, actor_depth * 0.06)
    semantic_visible = False
    semantic_pixels = 0
    semantic_ratio = 0.0
    if semantic_tags is not None:
        semantic_visible, semantic_pixels, semantic_ratio = bbox_has_vehicle_semantic_evidence(semantic_tags, bbox)

    visible_samples = 0
    for x, y, actor_point_depth in projected_in_frame:
        scene_depth = visible_depth_at(depth_meters, x, y)
        if scene_depth is None:
            continue
        if actor_point_depth <= scene_depth + tolerance_m and abs(scene_depth - actor_point_depth) <= tolerance_m * 1.8:
            visible_samples += 1

    depth_evidence_ratio = bbox_has_depth_evidence(depth_meters, bbox, actor_depth, tolerance_m * 1.5)
    min_visible_samples = 1 if (relaxed or area < 3000) else 2
    depth_ratio_floor = 0.004 if relaxed else 0.015
    if not semantic_visible and visible_samples < min_visible_samples and depth_evidence_ratio < depth_ratio_floor:
        return None, {
            "visibility_reject": "semantic_depth_occluded",
            "projected_bbox": bbox,
            "semantic_vehicle_pixels": semantic_pixels,
            "semantic_vehicle_ratio": round(semantic_ratio, 5),
            "visible_depth_samples": visible_samples,
            "depth_evidence_ratio": round(depth_evidence_ratio, 5),
            "actor_depth_m": round(actor_depth, 3),
        }

    visible_fraction = min(
        1.0,
        max(
            visible_samples / max(1, len(projected_in_frame)),
            depth_evidence_ratio * 8.0,
            semantic_ratio * 20.0,
        ),
    )
    return bbox, {
        "semantic_vehicle_pixels": semantic_pixels,
        "semantic_vehicle_ratio": round(semantic_ratio, 5),
        "visible_depth_samples": visible_samples,
        "projected_points_in_frame": len(projected_in_frame),
        "depth_evidence_ratio": round(depth_evidence_ratio, 5),
        "visibility_fraction": round(visible_fraction, 4),
        "actor_depth_m": round(actor_depth, 3),
    }


def parking_slot_corners(carla, slot_transform: object, length_m: float = 5.8, width_m: float = 2.8) -> list[object]:
    forward = yaw_to_forward(carla, slot_transform.rotation.yaw)
    right = yaw_to_right(carla, slot_transform.rotation.yaw)
    center = slot_transform.location
    corners = []
    for f_sign, r_sign in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
        corners.append(
            carla.Location(
                x=center.x + forward.x * f_sign * length_m * 0.5 + right.x * r_sign * width_m * 0.5,
                y=center.y + forward.y * f_sign * length_m * 0.5 + right.y * r_sign * width_m * 0.5,
                z=center.z + 0.08,
            )
        )
    return corners


def parking_lot_segments(carla, slots: dict[str, object]) -> list[tuple[object, object, str]]:
    segments: list[tuple[object, object, str]] = []
    for slot_id, slot_transform in slots.items():
        corners = parking_slot_corners(carla, slot_transform)
        for idx in range(len(corners)):
            segments.append((corners[idx], corners[(idx + 1) % len(corners)], slot_id))
    return segments


def draw_world_parking_lot_markings(carla, world: object, route: CarlaRoute, duration_sec: float) -> None:
    paint = carla.Color(176, 170, 150)
    for slots in (route.good_slots, route.defect_slots):
        for start, end, slot_id in parking_lot_segments(carla, slots):
            world.debug.draw_line(start, end, thickness=0.035, color=paint, life_time=duration_sec + 30.0)


def projected_line_points(
    start: object,
    end: object,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    x1, y1, d1 = get_image_point(start, intrinsic, world_to_camera)
    x2, y2, d2 = get_image_point(end, intrinsic, world_to_camera)
    if d1 <= 0.0 or d2 <= 0.0:
        return None
    return (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))


def blend_line(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color_bgr: tuple[int, int, int],
    thickness: int,
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.line(overlay, pt1, pt2, color_bgr, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def draw_worn_parking_line(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    seed: int,
    color_bgr: tuple[int, int, int] = (172, 182, 186),
) -> None:
    blend_line(frame, pt1, pt2, (35, 34, 31), 3, 0.08)
    rng = np.random.default_rng(seed)
    length = max(1.0, math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    segments = max(3, int(length / 34.0))
    for idx in range(segments):
        if rng.random() < 0.12:
            continue
        t0 = idx / segments + rng.uniform(0.0, 0.012)
        t1 = (idx + 1) / segments - rng.uniform(0.0, 0.018)
        if t1 <= t0:
            continue
        a = (
            int(round(pt1[0] + (pt2[0] - pt1[0]) * t0)),
            int(round(pt1[1] + (pt2[1] - pt1[1]) * t0)),
        )
        b = (
            int(round(pt1[0] + (pt2[0] - pt1[0]) * t1)),
            int(round(pt1[1] + (pt2[1] - pt1[1]) * t1)),
        )
        alpha = 0.34 + float(rng.uniform(-0.06, 0.05))
        blend_line(frame, a, b, color_bgr, 2, alpha)


def parking_wheel_stop_segments(carla, slots: dict[str, object]) -> list[tuple[object, object, str]]:
    segments: list[tuple[object, object, str]] = []
    for slot_id, slot_transform in slots.items():
        forward = yaw_to_forward(carla, slot_transform.rotation.yaw)
        right = yaw_to_right(carla, slot_transform.rotation.yaw)
        center = slot_transform.location
        stop_center = carla.Location(
            x=center.x + forward.x * 2.05,
            y=center.y + forward.y * 2.05,
            z=center.z + 0.10,
        )
        half_width = 0.88
        start = carla.Location(
            x=stop_center.x - right.x * half_width,
            y=stop_center.y - right.y * half_width,
            z=stop_center.z,
        )
        end = carla.Location(
            x=stop_center.x + right.x * half_width,
            y=stop_center.y + right.y * half_width,
            z=stop_center.z,
        )
        segments.append((start, end, slot_id))
    return segments


def overlay_projected_parking_lot(
    frame: np.ndarray,
    carla,
    slots: dict[str, object],
    camera_transform: object,
    intrinsic: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> None:
    world_to_camera = np.array(camera_transform.get_inverse_matrix())
    for idx, (start, end, slot_id) in enumerate(parking_lot_segments(carla, slots)):
        points = projected_line_points(start, end, intrinsic, world_to_camera)
        if points is None:
            continue
        seed = zlib.adler32(f"{slot_id}:{idx}".encode("utf-8")) & 0xFFFFFFFF
        draw_worn_parking_line(frame, points[0], points[1], seed)
    for idx, (start, end, slot_id) in enumerate(parking_wheel_stop_segments(carla, slots)):
        points = projected_line_points(start, end, intrinsic, world_to_camera)
        if points is None:
            continue
        seed = zlib.adler32(f"stop:{slot_id}:{idx}".encode("utf-8")) & 0xFFFFFFFF
        draw_worn_parking_line(frame, points[0], points[1], seed, (150, 160, 168))


def image_to_bgr(image: object) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    return array[:, :, :3].copy()


def cached_vignette(width: int, height: int) -> np.ndarray:
    key = (width, height)
    if key not in _VIGNETTE_CACHE:
        y, x = np.ogrid[:height, :width]
        cx = width * 0.5
        cy = height * 0.52
        distance = np.sqrt(((x - cx) / (width * 0.70)) ** 2 + ((y - cy) / (height * 0.70)) ** 2)
        vignette = 1.0 - np.clip((distance - 0.12) * 0.24, 0.0, 0.12)
        _VIGNETTE_CACHE[key] = vignette.astype(np.float32)[:, :, None]
    return _VIGNETTE_CACHE[key]


def cached_scanlines(width: int, height: int) -> np.ndarray:
    key = (width, height)
    if key not in _SCANLINE_CACHE:
        scanlines = np.ones((height, width, 1), dtype=np.float32)
        scanlines[1::4, :, :] = 0.985
        scanlines[3::8, :, :] = 0.992
        _SCANLINE_CACHE[key] = scanlines
    return _SCANLINE_CACHE[key]


def deterministic_frame_noise(shape: tuple[int, int, int], camera_id: str, frame_id: int, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    seed = (zlib.adler32(camera_id.encode("utf-8")) + frame_id * 2654435761) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, strength, shape).astype(np.float32)


def apply_filmic_curve(frame: np.ndarray) -> np.ndarray:
    normalized = np.clip(frame / 255.0, 0.0, 1.0)
    shoulder = normalized * (2.45 * normalized + 0.055) / (normalized * (2.43 * normalized + 0.59) + 0.14)
    return np.clip(shoulder * 255.0, 0.0, 255.0)


def apply_selective_bloom(frame: np.ndarray) -> np.ndarray:
    bright = np.maximum(frame - 185.0, 0.0) * 1.35
    bloom = cv2.GaussianBlur(np.clip(bright, 0, 255).astype(np.uint8), (0, 0), 8.0).astype(np.float32)
    return cv2.addWeighted(frame, 1.0, bloom, 0.10, 0.0)


def draw_cctv_overlay(frame: np.ndarray, camera_id: str, timestamp_sec: float, frame_id: int, fps: int) -> None:
    height, width = frame.shape[:2]
    panel_right = min(width - 18, 430)
    cv2.rectangle(frame, (18, 16), (panel_right, 62), (18, 22, 22), -1)
    cv2.rectangle(frame, (18, 16), (panel_right, 62), (52, 62, 62), 1, cv2.LINE_AA)
    cv2.circle(frame, (42, 39), 7, (72, 235, 125), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"REC  {camera_id}  T+{timestamp_sec:05.1f}s  {fps}FPS",
        (58, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (228, 235, 230),
        1,
        cv2.LINE_AA,
    )
    frame_label = f"FRAME {frame_id:05d}"
    (tw, _), _baseline = cv2.getTextSize(frame_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(
        frame,
        frame_label,
        (width - tw - 24, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (195, 205, 200),
        1,
        cv2.LINE_AA,
    )


def apply_aaa_cctv_postprocess(
    frame: np.ndarray,
    camera_id: str,
    timestamp_sec: float,
    frame_id: int,
    grain_strength: float,
    previous_frame: np.ndarray | None,
    temporal_blend_strength: float,
    fps: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    graded = frame.astype(np.float32)

    # AAA CCTV grade: filmic contrast, richer asphalt/vehicle separation, controlled highlights.
    graded = (graded - 128.0) * 1.13 + 128.0
    graded[:, :, 0] += 7.0
    graded[:, :, 1] += 2.5
    graded[:, :, 2] -= 2.0
    graded = apply_filmic_curve(graded)
    graded = graded * 1.08 + 5.0

    lab = cv2.cvtColor(np.clip(graded, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    graded = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR).astype(np.float32)

    blurred = cv2.GaussianBlur(np.clip(graded, 0, 255).astype(np.uint8), (0, 0), 1.2).astype(np.float32)
    graded = cv2.addWeighted(graded, 1.12, blurred, -0.12, 0.0)
    graded = apply_selective_bloom(graded)
    graded *= cached_vignette(width, height)
    graded *= cached_scanlines(width, height)
    graded += deterministic_frame_noise(frame.shape, camera_id, frame_id, grain_strength)

    output = np.clip(graded, 0, 255).astype(np.uint8)
    if previous_frame is not None and temporal_blend_strength > 0.0:
        output = cv2.addWeighted(output, 1.0 - temporal_blend_strength, previous_frame, temporal_blend_strength, 0.0)
    draw_cctv_overlay(output, camera_id, timestamp_sec, frame_id, fps)
    return output


def get_sensor_frame(sensor_queue: queue.Queue, frame: int, timeout_sec: float = 5.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            image = sensor_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if image.frame == frame:
            return image
        if image.frame > frame:
            raise RuntimeError(f"Camera queue skipped simulation frame {frame}; received {image.frame}.")
    raise RuntimeError(f"Timed out waiting for camera frame {frame}.")


def map_name_matches(current_map: str, requested_map: str) -> bool:
    return current_map.rsplit("/", 1)[-1] == requested_map


def resolve_carla_map(client: object, requested_map: str) -> str:
    available = client.get_available_maps()
    available_short = {name.rsplit("/", 1)[-1]: name for name in available}
    if requested_map in available_short:
        return requested_map
    if requested_map.endswith("_Opt"):
        fallback = requested_map.removesuffix("_Opt")
        if fallback in available_short:
            log_step(f"Map {requested_map} is not available; falling back to {fallback}")
            return fallback
    raise RuntimeError(
        f"CARLA map {requested_map} is not available. Available maps: "
        + ", ".join(sorted(available_short))
    )


def get_or_load_carla_map(client: object, requested_map: str) -> tuple[object, str]:
    resolved_map = resolve_carla_map(client, requested_map)
    world = client.get_world()
    current_map = world.get_map().name
    if map_name_matches(current_map, resolved_map):
        log_step(f"Using already loaded map: {current_map}")
        return world, resolved_map
    log_step(f"Loading {resolved_map} from current map {current_map}. This can take a while.")
    return client.load_world(resolved_map), resolved_map


def write_carla_route_plan(path: Path, route: CarlaRoute, carla_map: str) -> None:
    plan = {
        "dataset": "carla_honda_poc_aaa",
        "carla_map": carla_map,
        "route_flow": [
            "CAM_01_START",
            "CAM_02_TRANSIT",
            "CAM_03_JUNCTION_STATUS",
            "CAM_04_GOOD_ROUTE/CAM_05_DEFECT_ROUTE",
            "CAM_06_GOOD_PARKING/CAM_07_DEFECT_PARKING",
        ],
        "segment_windows_sec": SEGMENT_SECONDS,
        "route": {
            "common": [transform_to_dict(transform) for transform in route.common],
            "good": [transform_to_dict(transform) for transform in route.good],
            "defect": [transform_to_dict(transform) for transform in route.defect],
        },
        "parking_slots": {
            "good": {slot: transform_to_dict(transform) for slot, transform in route.good_slots.items()},
            "defect": {slot: transform_to_dict(transform) for slot, transform in route.defect_slots.items()},
        },
        "motion": {
            "spawn_cooldown_sec": "deterministic random 3.0-5.0",
            "parking_maneuver": "approach, overshoot, reverse entry, correction, final stop",
        },
        "validation": {
            "camera_overlap_required": True,
            "parking_final_transform_required": True,
            "parking_lot_markings": "muted worn parking paint and wheel stops on parking cameras",
            "render_strategy": "one_active_rgb_sensor_per_camera_with_matched_depth_and_semantic_visibility",
            "bbox_visibility": "3D projected boxes are written only when matched semantic/depth frames show visible vehicle evidence",
        },
    }
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def wait_for_tcp_port(host: str, port: int, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error: OSError | None = None
    progress = ProgressBar(f"waiting for TCP {host}:{port}", max(1, int(timeout_sec)))
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                progress.update(progress.total)
                return
        except OSError as exc:
            last_error = exc
            elapsed = int(timeout_sec - max(0.0, deadline - time.monotonic()))
            progress.update(elapsed, extra=str(exc))
            time.sleep(1.0)
    detail = f" Last socket error: {last_error}" if last_error else ""
    raise RuntimeError(
        f"CARLA server is not accepting TCP connections at {host}:{port} after "
        f"{timeout_sec:.0f}s.{detail} Start the CARLA Docker server and keep it running."
    )


def call_carla_rpc(description: str, fn):
    try:
        return fn()
    except RuntimeError as exc:
        raise RuntimeError(
            f"CARLA TCP port is open, but RPC call '{description}' timed out or failed. "
            "The simulator may still be loading, hung, or running in a bad state. "
            "Check the CARLA Docker terminal/logs and restart the CARLA container if needed. "
            f"Original error: {exc}"
        ) from exc


def configure_synchronous_settings(settings: object, fps: int) -> object:
    fixed_delta = 1.0 / fps
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta
    settings.no_rendering_mode = False

    if hasattr(settings, "substepping"):
        settings.substepping = True
    if hasattr(settings, "max_substep_delta_time") and hasattr(settings, "max_substeps"):
        target_substep = min(0.01, fixed_delta)
        substeps = max(1, math.ceil(fixed_delta / target_substep))
        if substeps > 10:
            substeps = 10
            target_substep = fixed_delta / substeps
        settings.max_substep_delta_time = target_substep
        settings.max_substeps = substeps
    return settings


def set_weather(carla, world: object) -> None:
    weather = carla.WeatherParameters()
    for attr, value in CONFIG.weather.items():
        if hasattr(weather, attr):
            setattr(weather, attr, value)
    world.set_weather(weather)


def distance_xy(a: object, b: object) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def distance_to_segment_xy(point: object, start: object, end: object) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-6:
        return distance_xy(point, start)
    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest = type(start)()
    closest.x = start.x + t * dx
    closest.y = start.y + t * dy
    closest.z = getattr(start, "z", 0.0)
    return distance_xy(point, closest)


def environment_object_location(obj: object) -> object | None:
    bounding_box = getattr(obj, "bounding_box", None)
    if bounding_box is not None and hasattr(bounding_box, "location"):
        return bounding_box.location
    transform = getattr(obj, "transform", None)
    if transform is not None and hasattr(transform, "location"):
        return transform.location
    return None


def maybe_get_city_label(carla, name: str):
    return getattr(carla.CityObjectLabel, name, None)


def hide_camera_blockers(carla, world: object, route: CarlaRoute, camera_transforms: dict[str, object]) -> set[int]:
    if not hasattr(world, "get_environment_objects") or not hasattr(world, "enable_environment_objects"):
        log_step("Environment object visibility toggling is not available on this CARLA server.")
        return set()

    blocker_segments = (
        (
            camera_transforms["CAM_02_TRANSIT"].location,
            route.common[min(16, len(route.common) - 1)].location,
            10.0,
        ),
        (
            camera_transforms["CAM_03_JUNCTION_STATUS"].location,
            route.common[-1].location,
            16.0,
        ),
    )
    hidden_ids: set[int] = set()
    for label_name in ("Buildings", "Fences", "Walls", "Vegetation", "Poles", "Other"):
        label = maybe_get_city_label(carla, label_name)
        if label is None:
            continue
        try:
            objects = world.get_environment_objects(label)
        except RuntimeError:
            continue
        for obj in objects:
            location = environment_object_location(obj)
            if location is None:
                continue
            blocks_view = any(
                distance_to_segment_xy(location, start, end) <= radius
                or distance_xy(location, start) <= radius * 0.75
                for start, end, radius in blocker_segments
            )
            if blocks_view:
                hidden_ids.add(obj.id)

    if hidden_ids:
        world.enable_environment_objects(hidden_ids, False)
        log_step(f"Temporarily hid {len(hidden_ids)} static environment objects near CAM_02/CAM_03.")
    else:
        log_step("No nearby CAM_02/CAM_03 static blockers were selected for hiding.")
    return hidden_ids


def spawn_carla_cameras(
    carla,
    world: object,
    camera_transforms: dict[str, object],
    camera_fovs: dict[str, float],
    width: int,
    height: int,
    fps: int,
) -> tuple[
    dict[str, object],
    dict[str, queue.Queue],
    dict[str, object],
    dict[str, queue.Queue],
    dict[str, object],
    dict[str, queue.Queue],
]:
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("sensor_tick", str(1.0 / fps))
    for attr, value in CONFIG.rgb_sensor_attributes.items():
        if camera_bp.has_attribute(attr):
            camera_bp.set_attribute(attr, value)

    depth_bp = blueprint_library.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(width))
    depth_bp.set_attribute("image_size_y", str(height))
    depth_bp.set_attribute("sensor_tick", str(1.0 / fps))

    semantic_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    semantic_bp.set_attribute("image_size_x", str(width))
    semantic_bp.set_attribute("image_size_y", str(height))
    semantic_bp.set_attribute("sensor_tick", str(1.0 / fps))

    cameras: dict[str, object] = {}
    queues: dict[str, queue.Queue] = {}
    depth_cameras: dict[str, object] = {}
    depth_queues: dict[str, queue.Queue] = {}
    semantic_cameras: dict[str, object] = {}
    semantic_queues: dict[str, queue.Queue] = {}
    for camera_id, transform in camera_transforms.items():
        fov = f"{camera_fovs.get(camera_id, 85.0):.1f}"
        camera_bp.set_attribute("fov", fov)
        depth_bp.set_attribute("fov", fov)
        semantic_bp.set_attribute("fov", fov)
        sensor = world.spawn_actor(camera_bp, transform)
        sensor_queue: queue.Queue = queue.Queue()
        sensor.listen(sensor_queue.put)
        depth_sensor = world.spawn_actor(depth_bp, transform)
        depth_queue: queue.Queue = queue.Queue()
        depth_sensor.listen(depth_queue.put)
        semantic_sensor = world.spawn_actor(semantic_bp, transform)
        semantic_queue: queue.Queue = queue.Queue()
        semantic_sensor.listen(semantic_queue.put)
        cameras[camera_id] = sensor
        queues[camera_id] = sensor_queue
        depth_cameras[camera_id] = depth_sensor
        depth_queues[camera_id] = depth_queue
        semantic_cameras[camera_id] = semantic_sensor
        semantic_queues[camera_id] = semantic_queue
    return cameras, queues, depth_cameras, depth_queues, semantic_cameras, semantic_queues


def spawn_carla_vehicles(carla, world: object, cars: list[CarSpec], route: CarlaRoute) -> dict[str, object]:
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter("vehicle.*")
    preferred = [
        bp for bp in vehicle_blueprints if any(wanted in bp.id for wanted in CONFIG.vehicle_blueprints)
    ]
    vehicle_bp = preferred[0] if preferred else vehicle_blueprints[0]
    actors: dict[str, object] = {}
    try:
        for idx, car in enumerate(cars):
            bp = vehicle_bp
            if bp.has_attribute("color"):
                b, g, r = car.color_bgr
                bp.set_attribute("color", f"{r},{g},{b}")

            path = path_for_car(carla, route, car)
            base_indices = [
                min(len(path) - 1, idx * 3),
                min(len(path) - 1, idx * 3 + 1),
                min(len(path) - 1, idx * 3 + 2),
                min(len(path) - 1, idx * 4),
            ]
            lateral_offsets = (0.0, 3.5, -3.5, 7.0, -7.0)
            z_offsets = (0.5, 1.5, 2.5)

            actor = None
            for base_idx in dict.fromkeys(base_indices):
                for lateral_m in lateral_offsets:
                    for z_m in z_offsets:
                        transform = clone_transform_with_offset(carla, path[base_idx], lateral_m, z_m)
                        actor = world.try_spawn_actor(bp, transform)
                        if actor is not None:
                            break
                    if actor is not None:
                        break
                if actor is not None:
                    break

            if actor is None:
                raise RuntimeError(
                    f"Could not spawn vehicle {car.tracking_id}; every candidate position collided."
                )

            actor.set_simulate_physics(False)
            actors[car.tracking_id] = actor
    except Exception:
        for actor in actors.values():
            try:
                actor.destroy()
            except RuntimeError:
                pass
        raise
    return actors


def write_carla_camera_graph(
    path: Path,
    camera_transforms: dict[str, object],
    camera_fovs: dict[str, float],
    width: int,
    height: int,
    fps: int,
    carla_version: str,
    carla_map: str,
) -> None:
    graph = {
        "dataset": "carla_honda_poc_aaa",
        "renderer": "carla",
        "visual_profile": {
            "preset": "aaa_cinematic_cctv",
            "weather": "low sun, cinematic shadows, mild atmospheric scattering",
            "postprocess": "filmic tone map, local contrast, bloom, cool CCTV grade, vignette, scanlines, deterministic grain, temporal stabilization",
            "parking_paint": "muted worn white parking stall lines with wheel stops; QA green/red parking overlays are disabled in this AAA profile",
            "vehicle_motion": "deterministic route transforms with velocity/angular-velocity hints for smoother rendered motion",
            "quality_note": "For maximum engine-side shadows and texture quality, start CARLA with -quality-level=Epic.",
        },
        "carla_map": carla_map,
        "carla_version": carla_version,
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "status_rule": {
            "camera_id": "CAM_03_JUNCTION_STATUS",
            "left_turn": "GOOD",
            "right_turn": "DEFECT",
        },
        "cameras": [
            {
                **asdict(camera),
                "camera_transform": transform_to_dict(camera_transforms[camera.camera_id]),
                "fov": camera_fovs[camera.camera_id],
            }
            for camera in CAMERAS
        ],
    }
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")


def write_carla_metadata(
    paths: dict[str, Path],
    cars: list[CarSpec],
    actors: dict[str, object],
    carla_version: str,
    carla_map: str,
) -> None:
    write_cars_csv(paths["metadata"] / "cars.csv", cars)
    with (paths["metadata"] / "events.jsonl").open("w", encoding="utf-8") as fh:
        for car in cars:
            for camera_id in car.route:
                start, end = segment_window(car, camera_id) or (0.0, 0.0)
                event = {
                    "tracking_id": car.tracking_id,
                    "status": car.status,
                    "camera_id": camera_id,
                    "vehicle_actor_id": actors[car.tracking_id].id,
                    "carla_map": carla_map,
                    "carla_version": carla_version,
                    "enter_timestamp_sec": round(start, 3),
                    "exit_timestamp_sec": round(end, 3),
                    "parking_slot_id": car.parking_slot_id if camera_id.endswith("PARKING") else "",
                }
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def render_carla(
    paths: dict[str, Path],
    cars: list[CarSpec],
    host: str,
    port: int,
    fps: int,
    duration_sec: float,
    width: int,
    height: int,
    carla_map: str,
    timeout_sec: float,
    cameras_to_render: list[CameraSpec],
    append_annotations: bool,
    hide_static_blockers: bool,
    cctv_postprocess: bool,
    cctv_grain_strength: float,
    temporal_stabilization: bool,
    temporal_blend_strength: float,
    wheel_motion_hints: bool,
    annotations_only: bool = False,
    min_bbox_px: int = 4,
) -> str:
    carla = import_carla_module()
    log_step(f"Connecting to CARLA at {host}:{port} with timeout {timeout_sec:.0f}s")
    wait_for_tcp_port(host, port, min(timeout_sec, 30.0))
    client = carla.Client(host, port)
    client.set_timeout(timeout_sec)
    carla_version = call_carla_rpc("get_client_version", client.get_client_version)
    if "0.9.15" not in carla_version:
        raise RuntimeError(f"Expected CARLA 0.9.15 client, got {carla_version}.")
    log_step("AAA cinematic visual profile enabled. For best shadows, run CARLA with -quality-level=Epic.")

    world, resolved_map = call_carla_rpc("get/load CARLA map", lambda: get_or_load_carla_map(client, carla_map))
    original_settings = world.get_settings()
    actors_to_destroy: list[object] = []
    writers: dict[str, cv2.VideoWriter] = {}
    hidden_environment_ids: set[int] = set()

    try:
        log_step("Configuring synchronous simulation")
        settings = world.get_settings()
        settings = configure_synchronous_settings(settings, fps)
        world.apply_settings(settings)
        set_weather(carla, world)

        log_step("Building route and camera transforms")
        route = build_carla_route(carla, world)
        camera_transforms = build_camera_transforms(carla, route)
        camera_fovs = carla_camera_fovs()
        if hide_static_blockers:
            hidden_environment_ids = hide_camera_blockers(carla, world, route, camera_transforms)
        draw_world_parking_lot_markings(carla, world, route, duration_sec)
        log_step(f"Spawning {len(cars)} vehicles")
        vehicle_actors = spawn_carla_vehicles(carla, world, cars, route)
        actors_to_destroy.extend(vehicle_actors.values())

        log_step("Writing CARLA metadata")
        write_carla_route_plan(paths["metadata"] / "route_plan.json", route, resolved_map)
        write_carla_camera_graph(
            paths["metadata"] / "camera_graph.json",
            camera_transforms,
            camera_fovs,
            width,
            height,
            fps,
            carla_version,
            resolved_map,
        )
        write_carla_metadata(paths, cars, vehicle_actors, carla_version, resolved_map)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        car_paths = {car.tracking_id: path_for_car(carla, route, car) for car in cars}
        frame_count = int(duration_sec * fps)
        bbox_path = paths["annotations"] / "bboxes.jsonl"

        projection_records = 0
        depth_filtered_records = 0
        route_window_records = 0
        bbox_mode = "a" if append_annotations and bbox_path.exists() else "w"
        with bbox_path.open(bbox_mode, encoding="utf-8") as bbox_fh:
            for camera in cameras_to_render:
                log_step(f"Rendering {camera.camera_id} with matched RGB/depth/semantic sensors")
                (
                    camera_actors,
                    camera_queues,
                    depth_actors,
                    depth_queues,
                    semantic_actors,
                    semantic_queues,
                ) = spawn_carla_cameras(
                    carla,
                    world,
                    {camera.camera_id: camera_transforms[camera.camera_id]},
                    camera_fovs,
                    width,
                    height,
                    fps,
                )
                camera_actor = camera_actors[camera.camera_id]
                camera_queue = camera_queues[camera.camera_id]
                depth_actor = depth_actors[camera.camera_id]
                depth_queue = depth_queues[camera.camera_id]
                semantic_actor = semantic_actors[camera.camera_id]
                semantic_queue = semantic_queues[camera.camera_id]
                writer = None
                if not annotations_only:
                    video_path = paths["videos"] / f"{camera.camera_id}.mp4"
                    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
                    if not writer.isOpened():
                        camera_actor.stop()
                        camera_actor.destroy()
                        depth_actor.stop()
                        depth_actor.destroy()
                        semantic_actor.stop()
                        semantic_actor.destroy()
                        raise RuntimeError(f"Could not open video writer: {video_path}")
                    writers[camera.camera_id] = writer

                try:
                    intrinsic = build_projection_matrix(width, height, camera_fovs[camera.camera_id])
                    warmup_frames = max(2, min(10, fps))
                    warmup = ProgressBar(f"warming {camera.camera_id}", warmup_frames)
                    for warmup_frame in range(warmup_frames):
                        set_vehicle_transforms_for_time(
                            carla,
                            cars,
                            vehicle_actors,
                            car_paths,
                            0.0,
                            duration_sec,
                            1.0 / fps,
                            wheel_motion_hints,
                        )
                        world.tick()
                        while not camera_queue.empty():
                            camera_queue.get_nowait()
                        while not depth_queue.empty():
                            depth_queue.get_nowait()
                        while not semantic_queue.empty():
                            semantic_queue.get_nowait()
                        warmup.update(warmup_frame + 1)

                    render_progress = ProgressBar(f"rendering {camera.camera_id}", frame_count)
                    previous_output_frame: np.ndarray | None = None
                    for local_frame_id in range(frame_count):
                        timestamp_sec = local_frame_id / fps
                        set_vehicle_transforms_for_time(
                            carla,
                            cars,
                            vehicle_actors,
                            car_paths,
                            timestamp_sec,
                            duration_sec,
                            1.0 / fps,
                            wheel_motion_hints,
                        )

                        snapshot = world.tick()
                        sim_frame = snapshot if isinstance(snapshot, int) else snapshot.frame
                        # Always drain the RGB/semantic queues to keep them frame-aligned, but only
                        # decode what we need. In annotations-only mode we skip the costly RGB decode,
                        # semantic decode, parking overlays, post-process and video encode.
                        if annotations_only:
                            get_sensor_frame(camera_queue, sim_frame)
                            depth_meters = depth_image_to_meters(get_sensor_frame(depth_queue, sim_frame))
                            get_sensor_frame(semantic_queue, sim_frame)
                            frame = None
                            semantic_tags = None
                        else:
                            frame = image_to_bgr(get_sensor_frame(camera_queue, sim_frame))
                            depth_meters = depth_image_to_meters(get_sensor_frame(depth_queue, sim_frame))
                            semantic_tags = semantic_image_to_tags(get_sensor_frame(semantic_queue, sim_frame))
                        camera_transform = camera_actor.get_transform()
                        if not annotations_only and camera.camera_id == "CAM_06_GOOD_PARKING":
                            overlay_projected_parking_lot(
                                frame, carla, route.good_slots, camera_transform, intrinsic, (70, 240, 120)
                            )
                        elif not annotations_only and camera.camera_id == "CAM_07_DEFECT_PARKING":
                            overlay_projected_parking_lot(
                                frame, carla, route.defect_slots, camera_transform, intrinsic, (90, 90, 255)
                            )

                        for car in cars:
                            if annotations_only:
                                # Annotate the car for its full in-frame duration: keep only the
                                # route-membership check and let the projection decide enter/exit.
                                if camera.camera_id not in car.route:
                                    continue
                            else:
                                visible_window = segment_window(car, camera.camera_id)
                                if visible_window is None:
                                    continue
                                visible_start, visible_end = visible_window
                                if timestamp_sec < visible_start or timestamp_sec > visible_end:
                                    continue
                            route_window_records += 1
                            actor = vehicle_actors[car.tracking_id]
                            bbox, visibility = yolo_like_visible_actor_bbox(
                                actor,
                                camera_transform,
                                intrinsic,
                                depth_meters,
                                semantic_tags,
                                width,
                                height,
                                min_bbox_px=min_bbox_px,
                                relaxed=annotations_only,
                            )
                            if not bbox:
                                depth_filtered_records += 1
                                continue
                            projection_records += 1
                            record = {
                                "tracking_id": car.tracking_id,
                                "status": car.status,
                                "route": list(car.route),
                                "camera_id": camera.camera_id,
                                "frame_id": local_frame_id,
                                "carla_frame": sim_frame,
                                "timestamp_sec": round(timestamp_sec, 3),
                                "bbox": bbox,
                                "bbox_source": "carla_3d_projection_depth_visible",
                                "visibility": visibility,
                                "parking_slot_id": car.parking_slot_id if camera.camera_id.endswith("PARKING") else "",
                                "vehicle_actor_id": actor.id,
                                "camera_transform": transform_to_dict(camera_transform),
                                "world_transform": transform_to_dict(actor.get_transform()),
                            }
                            bbox_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if not annotations_only:
                            if cctv_postprocess:
                                frame = apply_aaa_cctv_postprocess(
                                    frame,
                                    camera.camera_id,
                                    timestamp_sec,
                                    local_frame_id,
                                    cctv_grain_strength,
                                    previous_output_frame if temporal_stabilization else None,
                                    temporal_blend_strength if temporal_stabilization else 0.0,
                                    fps,
                                )
                                previous_output_frame = frame.copy()
                            writer.write(frame)
                        render_progress.update(
                            local_frame_id + 1,
                            extra=f"sim_frame={sim_frame}",
                        )
                finally:
                    if writer is not None:
                        writer.release()
                    writers.pop(camera.camera_id, None)
                    try:
                        camera_actor.stop()
                        camera_actor.destroy()
                        depth_actor.stop()
                        depth_actor.destroy()
                        semantic_actor.stop()
                        semantic_actor.destroy()
                        world.tick()
                    except RuntimeError:
                        pass
        log_step(
            "Annotation records written: "
            f"{projection_records} depth-visible projected, "
            f"{depth_filtered_records} depth-filtered from {route_window_records} route-window candidates"
        )
        if projection_records == 0:
            raise RuntimeError(
                "CARLA camera projection produced 0 bounding boxes. "
                "Camera placement is likely wrong; inspect generated videos/contact sheets."
            )
    finally:
        log_step("Cleaning up CARLA actors and restoring settings")
        for writer in writers.values():
            writer.release()
        for actor in actors_to_destroy:
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        if hidden_environment_ids:
            try:
                world.enable_environment_objects(hidden_environment_ids, True)
            except RuntimeError:
                pass
        world.apply_settings(original_settings)

    return carla_version


def validate_carla_connection(host: str, port: int) -> None:
    carla = import_carla_module()

    client = carla.Client(host, port)
    client.set_timeout(10.0)
    version = client.get_client_version()
    world, resolved_map = get_or_load_carla_map(client, "Town05_Opt")
    world.wait_for_tick()
    print(f"Connected to CARLA client {version}; loaded {world.get_map().name} ({resolved_map}).")


def draw_map_png(path: Path) -> None:
    width, height = 1220, 540
    img = Image.new("RGB", (width, height), (246, 248, 250))
    draw = ImageDraw.Draw(img)
    title_font = font(26)
    label_font = font(17)
    small_font = font(13)

    draw.text((34, 26), "Honda Smart Car Tracking POC - Camera Map", font=title_font, fill=(24, 32, 42))

    edges = [
        ("CAM_01_START", "CAM_02_TRANSIT"),
        ("CAM_02_TRANSIT", "CAM_03_JUNCTION_STATUS"),
        ("CAM_03_JUNCTION_STATUS", "CAM_04_GOOD_ROUTE"),
        ("CAM_03_JUNCTION_STATUS", "CAM_05_DEFECT_ROUTE"),
        ("CAM_04_GOOD_ROUTE", "CAM_06_GOOD_PARKING"),
        ("CAM_05_DEFECT_ROUTE", "CAM_07_DEFECT_PARKING"),
    ]
    camera_by_id = {camera.camera_id: camera for camera in CAMERAS}

    for start_id, end_id in edges:
        start = camera_by_id[start_id].map_xy
        end = camera_by_id[end_id].map_xy
        color = (44, 130, 201)
        if "GOOD" in end_id:
            color = (30, 145, 95)
        if "DEFECT" in end_id:
            color = (205, 82, 82)
        draw.line((start, end), fill=color, width=7)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        arrow_x = end[0] - 32 * math.cos(angle)
        arrow_y = end[1] - 32 * math.sin(angle)
        left = (arrow_x - 12 * math.cos(angle - 0.8), arrow_y - 12 * math.sin(angle - 0.8))
        right = (arrow_x - 12 * math.cos(angle + 0.8), arrow_y - 12 * math.sin(angle + 0.8))
        draw.polygon((end, left, right), fill=color)

    for camera in CAMERAS:
        x, y = camera.map_xy
        fill = (255, 255, 255)
        outline = (44, 130, 201)
        if "GOOD" in camera.camera_id:
            outline = (30, 145, 95)
        if "DEFECT" in camera.camera_id:
            outline = (205, 82, 82)
        draw.rounded_rectangle((x - 98, y - 34, x + 98, y + 34), radius=8, fill=fill, outline=outline, width=4)
        draw.text((x - 84, y - 20), camera.camera_id.replace("CAM_", "C"), font=small_font, fill=(20, 28, 36))
        draw.text((x - 84, y + 2), camera.name, font=small_font, fill=(20, 28, 36))

    draw.rounded_rectangle((40, 420, 1175, 500), radius=8, fill=(255, 255, 255), outline=(210, 216, 222))
    draw.text((60, 436), "Status rule at CAM_03_JUNCTION_STATUS", font=label_font, fill=(24, 32, 42))
    draw.text((60, 466), "Left turn = GOOD -> G01-G06", font=label_font, fill=(30, 145, 95))
    draw.text((390, 466), "Right turn = DEFECT -> D01-D04", font=label_font, fill=(205, 82, 82))
    img.save(path)


def write_dataset_doc(path: Path, output_dir: Path) -> None:
    text = f"""# CARLA Honda POC Dataset

This dataset supports the Honda Smart Car Tracking System POC described in
`Proposal_v1_0_Honda_Smart_Car_Tracking_System.md`.

The final dataset must be generated with `--renderer carla`. The storyboard
renderer is only a lightweight dry-run for validating metadata and camera graph
logic on machines that do not have CARLA installed.

## Scenario

- CARLA target map: `Town05_Opt` with fallback to `Town05`
- Dataset scale: Focused POC, 5-6 vehicles
- Cameras: 7 fixed virtual CCTV viewpoints rendered by CARLA RGB sensors
- Status rule: at `CAM_03_JUNCTION_STATUS`, left turn is `GOOD` and right turn is `DEFECT`
- Parking slots: `G01-G06` for GOOD vehicles and `D01-D04` for DEFECT vehicles
- Vehicle spawn cooldown: deterministic random 3-5 seconds
- Parking behavior: approach, overshoot, reverse entry, correction, final stop
- Parking lot markings: projected slot lines on parking camera videos
- Bounding boxes: CARLA 3D vehicle boxes projected into each camera plane, then filtered by matched semantic/depth visibility so hidden/occluded vehicles are not annotated
- OCR is disabled for this POC iteration while vehicle tracking and parking are validated.

## Generated Files

- `{output_dir}/videos/*.mp4`
- `{output_dir}/metadata/cars.csv`
- `{output_dir}/metadata/camera_graph.json`
- `{output_dir}/metadata/route_plan.json` (CARLA renderer)
- `{output_dir}/metadata/events.jsonl`
- `{output_dir}/annotations/bboxes.jsonl`
- `docs/carla_honda_poc_map.png`

## Generate Locally

Run CARLA 0.9.15 first, then generate the AAA-style 3D CCTV dataset:

```bash
python scripts/generate_carla_dataset_aaa.py --clean --write-contact-sheets
```

For metadata-only development without CARLA, use the storyboard dry-run:

```bash
python scripts/generate_carla_dataset_aaa.py --renderer storyboard --clean
```
"""
    path.write_text(text, encoding="utf-8")


def write_video_contact_sheets(output_dir: Path, docs_dir: Path) -> None:
    sheet_dir = docs_dir / "video_contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for video_path in sorted((output_dir / "videos").glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        if total <= 0:
            cap.release()
            continue
        sample_ids = [
            0,
            int(total * 0.15),
            int(total * 0.30),
            int(total * 0.45),
            int(total * 0.60),
            int(total * 0.75),
            total - 1,
        ]
        frames: list[np.ndarray] = []
        for frame_id in sample_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (320, 180))
            label = f"{video_path.stem} f={frame_id} t={frame_id / fps:.1f}s"
            cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(frame)
        cap.release()
        if frames:
            cv2.imwrite(str(sheet_dir / f"{video_path.stem}.jpg"), np.vstack(frames))


def validate_outputs(
    output_dir: Path,
    cars: list[CarSpec],
    expected_camera_ids: Iterable[str],
    require_route_plan: bool = False,
) -> None:
    del cars
    expected_camera_ids = tuple(expected_camera_ids)
    videos = sorted((output_dir / "videos").glob("*.mp4"))
    video_ids = {video.stem for video in videos}
    missing_videos = [camera_id for camera_id in expected_camera_ids if camera_id not in video_ids]
    if missing_videos:
        raise RuntimeError(f"Missing videos for cameras: {', '.join(missing_videos)}.")

    required = [
        output_dir / "metadata" / "cars.csv",
        output_dir / "metadata" / "camera_graph.json",
        output_dir / "metadata" / "events.jsonl",
        output_dir / "annotations" / "bboxes.jsonl",
    ]
    if require_route_plan:
        required.append(output_dir / "metadata" / "route_plan.json")
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")

    with (output_dir / "metadata" / "camera_graph.json").open(encoding="utf-8") as fh:
        graph = json.load(fh)
    if len(graph.get("cameras", [])) != len(CAMERAS):
        raise RuntimeError(f"Expected {len(CAMERAS)} cameras in camera_graph.json.")


def apply_preset_defaults(args: argparse.Namespace, cfg: DatasetConfig) -> None:
    """Fill any flag left at its None sentinel from the active preset."""
    for field_name in (
        "output_dir",
        "docs_dir",
        "num_cars",
        "fps",
        "duration_sec",
        "width",
        "height",
        "carla_map",
        "random_seed",
        "cctv_grain_strength",
        "temporal_blend_strength",
    ):
        if getattr(args, field_name) is None:
            setattr(args, field_name, getattr(cfg, field_name))


def main() -> int:
    global CONFIG
    args = parse_args()
    CONFIG = get_preset(args.preset)
    apply_preset_defaults(args, CONFIG)
    validate_args(args)

    output_dir = Path(args.output_dir)
    docs_dir = Path(args.docs_dir)
    paths = ensure_dirs(output_dir, docs_dir, args.clean)
    cars = make_cars(args.num_cars, args.random_seed)
    cameras_to_render = selected_cameras(args.camera_ids)

    if args.renderer == "carla":
        log_step("Selected cameras: " + ", ".join(camera.camera_id for camera in cameras_to_render))
        carla_version = render_carla(
            paths,
            cars,
            args.carla_host,
            args.carla_port,
            args.fps,
            args.duration_sec,
            args.width,
            args.height,
            args.carla_map,
            args.carla_timeout_sec,
            cameras_to_render,
            args.append_annotations,
            args.hide_camera_blockers,
            args.cctv_postprocess,
            args.cctv_grain_strength,
            args.temporal_stabilization,
            args.temporal_blend_strength,
            args.wheel_motion_hints,
            args.annotations_only,
            args.min_bbox_px,
        )
        if args.annotations_only:
            print(f"Re-projected bbox annotations with CARLA {carla_version} (videos untouched).")
        else:
            print(f"Generated AAA-style CARLA 3D CCTV videos with CARLA {carla_version}.")
    else:
        write_camera_graph(paths["metadata"] / "camera_graph.json")
        write_cars_csv(paths["metadata"] / "cars.csv", cars)
        write_events(paths["metadata"] / "events.jsonl", cars)
        render_storyboard(paths, cars, args.fps, args.duration_sec, args.width, args.height)
        print("Generated storyboard dry-run videos. Use --renderer carla for the AAA-style 3D CCTV dataset.")

    draw_map_png(docs_dir / "carla_honda_poc_map.png")
    write_dataset_doc(docs_dir / "carla_honda_poc_dataset.md", output_dir)
    validate_outputs(
        output_dir,
        cars,
        [camera.camera_id for camera in cameras_to_render],
        require_route_plan=args.renderer == "carla",
    )
    if args.write_contact_sheets:
        write_video_contact_sheets(output_dir, docs_dir)
        print(f"Contact sheets written to {docs_dir / 'video_contact_sheets'}")

    print(f"Generated {len(CAMERAS)} videos for {len(cars)} cars in {output_dir}")
    print(f"Map written to {docs_dir / 'carla_honda_poc_map.png'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
