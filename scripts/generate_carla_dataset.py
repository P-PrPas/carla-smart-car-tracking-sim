#!/usr/bin/env python3
"""Generate the Honda Smart Car Tracking POC dataset.

The generator has two renderers:
- storyboard: creates deterministic MP4 videos and metadata without CARLA.
- carla: validates a local CARLA 0.9.15 connection, then uses the same
  scenario spec. This keeps the repo runnable on machines without CARLA while
  preserving the CARLA-oriented dataset contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CAMERA_IDS = [
    "CAM_01_START_OCR",
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
    oil_tank_id: str
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
        "CAM_01_START_OCR",
        "Start OCR",
        "Vehicle identification and six-digit oil tank ID OCR.",
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
    "CAM_01_START_OCR": (0.0, 8.0),
    "CAM_02_TRANSIT": (7.0, 15.0),
    "CAM_03_JUNCTION_STATUS": (14.0, 23.0),
    "CAM_04_GOOD_ROUTE": (22.0, 31.0),
    "CAM_05_DEFECT_ROUTE": (22.0, 31.0),
    "CAM_06_GOOD_PARKING": (30.0, 42.0),
    "CAM_07_DEFECT_PARKING": (30.0, 42.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Honda Smart Car Tracking POC video dataset."
    )
    parser.add_argument("--output-dir", default="datasets/carla_honda_poc")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--renderer", choices=("storyboard", "carla"), default="storyboard")
    parser.add_argument("--num-cars", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration-sec", type=float, default=45.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--oil-start", type=int, default=100001)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--clean", action="store_true", help="Remove previous generated outputs.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 8 <= args.num_cars <= 12:
        raise ValueError("--num-cars must be between 8 and 12 for the Small POC dataset.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.duration_sec < 30:
        raise ValueError("--duration-sec must be at least 30 seconds.")
    if args.width < 640 or args.height < 360:
        raise ValueError("--width/--height are too small for readable OCR preview.")


def make_cars(num_cars: int, oil_start: int) -> list[CarSpec]:
    colors = [
        (210, 210, 205),
        (190, 195, 200),
        (225, 225, 220),
        (175, 185, 195),
        (205, 205, 215),
        (185, 190, 185),
        (220, 215, 205),
        (200, 205, 210),
        (170, 180, 190),
        (230, 230, 225),
        (195, 198, 205),
        (212, 216, 218),
    ]
    cars: list[CarSpec] = []
    good_slots = ["G01", "G02", "G03", "G04", "G05", "G06"]
    defect_slots = ["D01", "D02", "D03", "D04"]

    for idx in range(num_cars):
        status = "GOOD" if idx % 3 != 1 else "DEFECT"
        if status == "GOOD":
            route = (
                "CAM_01_START_OCR",
                "CAM_02_TRANSIT",
                "CAM_03_JUNCTION_STATUS",
                "CAM_04_GOOD_ROUTE",
                "CAM_06_GOOD_PARKING",
            )
            slot = good_slots[(idx // 2) % len(good_slots)]
        else:
            route = (
                "CAM_01_START_OCR",
                "CAM_02_TRANSIT",
                "CAM_03_JUNCTION_STATUS",
                "CAM_05_DEFECT_ROUTE",
                "CAM_07_DEFECT_PARKING",
            )
            slot = defect_slots[(idx // 3) % len(defect_slots)]

        oil_tank_id = f"{oil_start + idx:06d}"
        if not re.fullmatch(r"\d{6}", oil_tank_id):
            raise ValueError(f"oil_tank_id must be six digits: {oil_tank_id}")

        cars.append(
            CarSpec(
                tracking_id=f"TRK_{idx + 1:04d}",
                oil_tank_id=oil_tank_id,
                status=status,
                route=route,
                parking_slot_id=slot,
                color_bgr=colors[idx % len(colors)],
                start_offset_sec=idx * 2.2,
                speed_factor=0.92 + (idx % 5) * 0.04,
            )
        )
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
        "dataset": "carla_honda_poc",
        "renderer": "storyboard",
        "carla_map": "Town05",
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
                "oil_tank_id",
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
                    "oil_tank_id": car.oil_tank_id,
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
        base_start + car.start_offset_sec,
        base_end + car.start_offset_sec / car.speed_factor,
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

    if camera_id == "CAM_01_START_OCR":
        tag_x1 = x1 + int((x2 - x1) * 0.23)
        tag_x2 = x2 - int((x2 - x1) * 0.23)
        tag_y1 = y1 + 9
        tag_y2 = tag_y1 + max(18, int((y2 - y1) * 0.28))
        draw.rectangle((tag_x1, tag_y1, tag_x2, tag_y2), fill=(255, 255, 245), outline=(40, 40, 40))
        text_font = font(max(16, int((tag_y2 - tag_y1) * 0.74)))
        text_box = draw.textbbox((0, 0), car.oil_tank_id, font=text_font)
        tx = tag_x1 + ((tag_x2 - tag_x1) - (text_box[2] - text_box[0])) // 2
        ty = tag_y1 + ((tag_y2 - tag_y1) - (text_box[3] - text_box[1])) // 2 - 1
        draw.text((tx, ty), car.oil_tank_id, font=text_font, fill=(10, 10, 10))

    label = f"{car.tracking_id} {car.status}"
    draw.text((x1, max(4, y1 - 22)), label, fill=(255, 255, 255))


def write_events(path: Path, cars: Iterable[CarSpec]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for car in cars:
            for camera_id in car.route:
                start, end = segment_window(car, camera_id) or (0.0, 0.0)
                event = {
                    "tracking_id": car.tracking_id,
                    "oil_tank_id": car.oil_tank_id,
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
                        "oil_tank_id": car.oil_tank_id,
                        "status": car.status,
                        "route": list(car.route),
                        "camera_id": camera.camera_id,
                        "frame_id": frame_id,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "bbox": bbox,
                        "bbox_source": "storyboard_2d",
                        "ocr_bbox": bbox if camera.camera_id == "CAM_01_START_OCR" else None,
                        "parking_slot_id": car.parking_slot_id if camera.camera_id.endswith("PARKING") else "",
                    }
                    bbox_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
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
    pitch = -math.degrees(math.atan2(dz, distance_xy))
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


def build_carla_route(carla, world: object) -> CarlaRoute:
    world_map = world.get_map()
    spawn_points = sorted(
        world_map.get_spawn_points(),
        key=lambda transform: (round(transform.location.x, 2), round(transform.location.y, 2)),
    )
    if not spawn_points:
        raise RuntimeError("Town05 returned no spawn points.")

    common = sample_forward_route(carla, world_map, spawn_points[0], count=24, step_m=4.0)
    junction = common[14]
    forward = yaw_to_forward(carla, junction.rotation.yaw)
    right = yaw_to_right(carla, junction.rotation.yaw)

    def branch(sign: float, count: int) -> list[object]:
        transforms = []
        for idx in range(count):
            dist = 5.0 + idx * 4.0
            lateral = sign * min(18.0, 3.0 + idx * 2.2)
            loc = carla.Location(
                x=junction.location.x + forward.x * dist + right.x * lateral,
                y=junction.location.y + forward.y * dist + right.y * lateral,
                z=junction.location.z,
            )
            transforms.append(project_to_road(carla, world_map, loc))
        return transforms

    good = branch(-1.0, 18)
    defect = branch(1.0, 18)

    def slots(branch_transforms: list[object], prefix: str, count: int, sign: float) -> dict[str, object]:
        base = branch_transforms[-1]
        base_forward = yaw_to_forward(carla, base.rotation.yaw)
        base_right = yaw_to_right(carla, base.rotation.yaw)
        result = {}
        for idx in range(count):
            loc = carla.Location(
                x=base.location.x + base_forward.x * (idx * 2.5) + base_right.x * sign * (4.0 + idx * 1.1),
                y=base.location.y + base_forward.y * (idx * 2.5) + base_right.y * sign * (4.0 + idx * 1.1),
                z=base.location.z,
            )
            slot_transform = project_to_road(carla, world_map, loc)
            slot_transform.rotation.yaw = base.rotation.yaw + sign * 72.0
            result[f"{prefix}{idx + 1:02d}"] = slot_transform
        return result

    return CarlaRoute(
        common=tuple(common),
        good=tuple(good),
        defect=tuple(defect),
        good_slots=slots(good, "G", 6, -1.0),
        defect_slots=slots(defect, "D", 4, 1.0),
    )


def path_for_car(route: CarlaRoute, car: CarSpec) -> tuple[object, ...]:
    if car.status == "GOOD":
        return tuple(route.common[:15]) + route.good + (route.good_slots[car.parking_slot_id],)
    return tuple(route.common[:15]) + route.defect + (route.defect_slots[car.parking_slot_id],)


def route_transform_at(carla, transforms: tuple[object, ...], progress: float):
    progress = max(0.0, min(1.0, progress))
    if len(transforms) == 1:
        return transforms[0]
    scaled = progress * (len(transforms) - 1)
    idx = min(len(transforms) - 2, int(scaled))
    alpha = scaled - idx
    return make_transform_between(carla, transforms[idx].location, transforms[idx + 1].location, alpha)


def camera_targets_from_route(route: CarlaRoute) -> dict[str, object]:
    return {
        "CAM_01_START_OCR": route.common[2].location,
        "CAM_02_TRANSIT": route.common[8].location,
        "CAM_03_JUNCTION_STATUS": route.common[14].location,
        "CAM_04_GOOD_ROUTE": route.good[7].location,
        "CAM_05_DEFECT_ROUTE": route.defect[7].location,
        "CAM_06_GOOD_PARKING": route.good[-1].location,
        "CAM_07_DEFECT_PARKING": route.defect[-1].location,
    }


def build_camera_transforms(carla, route: CarlaRoute) -> dict[str, object]:
    targets = camera_targets_from_route(route)
    transforms: dict[str, object] = {}
    for camera in CAMERAS:
        target = targets[camera.camera_id]
        if "GOOD" in camera.camera_id:
            offset_yaw = -135.0
        elif "DEFECT" in camera.camera_id:
            offset_yaw = 135.0
        elif camera.camera_id == "CAM_03_JUNCTION_STATUS":
            offset_yaw = 180.0
        else:
            offset_yaw = -160.0
        reference_yaw = route.common[min(10, len(route.common) - 1)].rotation.yaw + offset_yaw
        offset = yaw_to_forward(carla, reference_yaw)
        location = carla.Location(
            x=target.x + offset.x * 22.0,
            y=target.y + offset.y * 22.0,
            z=target.z + 9.0,
        )
        transforms[camera.camera_id] = carla.Transform(location, look_at_rotation(carla, location, target))
    return transforms


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


def project_actor_bbox(
    actor: object,
    camera_transform: object,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> list[int] | None:
    world_to_camera = np.array(camera_transform.get_inverse_matrix())
    vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
    projected = [get_image_point(vertex, intrinsic, world_to_camera) for vertex in vertices]
    visible = [(x, y) for x, y, depth in projected if depth > 0.0]
    if len(visible) < 4:
        return None
    xs = [point[0] for point in visible]
    ys = [point[1] for point in visible]
    x1 = max(0, int(math.floor(min(xs))))
    y1 = max(0, int(math.floor(min(ys))))
    x2 = min(width - 1, int(math.ceil(max(xs))))
    y2 = min(height - 1, int(math.ceil(max(ys))))
    if x2 <= x1 or y2 <= y1:
        return None
    if x2 < 0 or y2 < 0 or x1 >= width or y1 >= height:
        return None
    return [x1, y1, x2, y2]


def projected_oil_tag_bbox(car_bbox: list[int]) -> list[int] | None:
    x1, y1, x2, y2 = car_bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw < 80 or bh < 40:
        return None
    tag_w = int(bw * 0.34)
    tag_h = int(max(18, bh * 0.18))
    cx = int((x1 + x2) / 2)
    tag_x1 = max(0, cx - tag_w // 2)
    tag_y1 = max(0, y1 + int(bh * 0.18))
    return [tag_x1, tag_y1, tag_x1 + tag_w, tag_y1 + tag_h]


def overlay_oil_tag(frame_bgr: np.ndarray, oil_tank_id: str, tag_bbox: list[int]) -> None:
    x1, y1, x2, y2 = tag_bbox
    x2 = min(frame_bgr.shape[1] - 1, x2)
    y2 = min(frame_bgr.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (245, 245, 255), thickness=-1)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (20, 20, 20), thickness=1)
    font_scale = max(0.45, min(1.25, (y2 - y1) / 28.0))
    thickness = max(1, int(round(font_scale * 2)))
    text_size, _ = cv2.getTextSize(oil_tank_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = x1 + max(2, ((x2 - x1) - text_size[0]) // 2)
    ty = y1 + max(text_size[1] + 2, ((y2 - y1) + text_size[1]) // 2)
    cv2.putText(
        frame_bgr,
        oil_tank_id,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (5, 5, 5),
        thickness,
        cv2.LINE_AA,
    )


def image_to_bgr(image: object) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    return array[:, :, :3].copy()


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


def set_weather(carla, world: object) -> None:
    weather = carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        sun_altitude_angle=55.0,
        sun_azimuth_angle=35.0,
        fog_density=0.0,
        wetness=0.0,
    )
    world.set_weather(weather)


def spawn_carla_cameras(
    carla,
    world: object,
    camera_transforms: dict[str, object],
    width: int,
    height: int,
    fps: int,
) -> tuple[dict[str, object], dict[str, queue.Queue]]:
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", "70")
    camera_bp.set_attribute("sensor_tick", str(1.0 / fps))
    for attr, value in {
        "enable_postprocess_effects": "True",
        "exposure_mode": "manual",
        "exposure_compensation": "0",
        "gamma": "2.2",
        "motion_blur_intensity": "0.0",
    }.items():
        if camera_bp.has_attribute(attr):
            camera_bp.set_attribute(attr, value)

    cameras: dict[str, object] = {}
    queues: dict[str, queue.Queue] = {}
    for camera_id, transform in camera_transforms.items():
        sensor = world.spawn_actor(camera_bp, transform)
        sensor_queue: queue.Queue = queue.Queue()
        sensor.listen(sensor_queue.put)
        cameras[camera_id] = sensor
        queues[camera_id] = sensor_queue
    return cameras, queues


def spawn_carla_vehicles(carla, world: object, cars: list[CarSpec], route: CarlaRoute) -> dict[str, object]:
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter("vehicle.*")
    preferred = [
        bp for bp in vehicle_blueprints if "vehicle.lincoln.mkz_2020" in bp.id or "vehicle.tesla.model3" in bp.id
    ]
    vehicle_bp = preferred[0] if preferred else vehicle_blueprints[0]
    actors: dict[str, object] = {}
    for idx, car in enumerate(cars):
        bp = vehicle_bp
        if bp.has_attribute("color"):
            b, g, r = car.color_bgr
            bp.set_attribute("color", f"{r},{g},{b}")
        transform = path_for_car(route, car)[0]
        transform.location.z += 0.2 + idx * 0.005
        actor = world.spawn_actor(bp, transform)
        actor.set_simulate_physics(False)
        actors[car.tracking_id] = actor
    return actors


def write_carla_camera_graph(
    path: Path,
    camera_transforms: dict[str, object],
    width: int,
    height: int,
    fps: int,
    carla_version: str,
) -> None:
    graph = {
        "dataset": "carla_honda_poc",
        "renderer": "carla",
        "carla_map": "Town05",
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
                "fov": 70.0,
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
) -> None:
    write_cars_csv(paths["metadata"] / "cars.csv", cars)
    with (paths["metadata"] / "events.jsonl").open("w", encoding="utf-8") as fh:
        for car in cars:
            for camera_id in car.route:
                start, end = segment_window(car, camera_id) or (0.0, 0.0)
                event = {
                    "tracking_id": car.tracking_id,
                    "oil_tank_id": car.oil_tank_id,
                    "status": car.status,
                    "camera_id": camera_id,
                    "vehicle_actor_id": actors[car.tracking_id].id,
                    "carla_map": "Town05",
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
) -> str:
    carla = import_carla_module()
    client = carla.Client(host, port)
    client.set_timeout(20.0)
    carla_version = client.get_client_version()
    if "0.9.15" not in carla_version:
        raise RuntimeError(f"Expected CARLA 0.9.15 client, got {carla_version}.")

    world = client.load_world("Town05")
    original_settings = world.get_settings()
    actors_to_destroy: list[object] = []
    writers: dict[str, cv2.VideoWriter] = {}

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / fps
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        set_weather(carla, world)

        route = build_carla_route(carla, world)
        camera_transforms = build_camera_transforms(carla, route)
        camera_actors, camera_queues = spawn_carla_cameras(carla, world, camera_transforms, width, height, fps)
        vehicle_actors = spawn_carla_vehicles(carla, world, cars, route)
        actors_to_destroy.extend(camera_actors.values())
        actors_to_destroy.extend(vehicle_actors.values())

        write_carla_camera_graph(
            paths["metadata"] / "camera_graph.json",
            camera_transforms,
            width,
            height,
            fps,
            carla_version,
        )
        write_carla_metadata(paths, cars, vehicle_actors, carla_version)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for camera in CAMERAS:
            video_path = paths["videos"] / f"{camera.camera_id}.mp4"
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {video_path}")
            writers[camera.camera_id] = writer

        car_paths = {car.tracking_id: path_for_car(route, car) for car in cars}
        intrinsic = build_projection_matrix(width, height, 70.0)
        frame_count = int(duration_sec * fps)
        bbox_path = paths["annotations"] / "bboxes.jsonl"

        with bbox_path.open("w", encoding="utf-8") as bbox_fh:
            for local_frame_id in range(frame_count):
                timestamp_sec = local_frame_id / fps
                for car in cars:
                    start = car.start_offset_sec
                    end = max(start + 1.0, duration_sec - 2.0 + car.start_offset_sec * 0.05)
                    progress = (timestamp_sec - start) / (end - start)
                    actor = vehicle_actors[car.tracking_id]
                    actor.set_transform(route_transform_at(carla, car_paths[car.tracking_id], progress))

                snapshot = world.tick()
                sim_frame = snapshot if isinstance(snapshot, int) else snapshot.frame
                frames = {
                    camera.camera_id: image_to_bgr(get_sensor_frame(camera_queues[camera.camera_id], sim_frame))
                    for camera in CAMERAS
                }

                for camera in CAMERAS:
                    camera_transform = camera_actors[camera.camera_id].get_transform()
                    frame = frames[camera.camera_id]
                    for car in cars:
                        actor = vehicle_actors[car.tracking_id]
                        bbox = project_actor_bbox(actor, camera_transform, intrinsic, width, height)
                        if not bbox:
                            continue
                        ocr_bbox = None
                        if camera.camera_id == "CAM_01_START_OCR":
                            ocr_bbox = projected_oil_tag_bbox(bbox)
                            if ocr_bbox:
                                overlay_oil_tag(frame, car.oil_tank_id, ocr_bbox)
                        record = {
                            "tracking_id": car.tracking_id,
                            "oil_tank_id": car.oil_tank_id,
                            "status": car.status,
                            "route": list(car.route),
                            "camera_id": camera.camera_id,
                            "frame_id": local_frame_id,
                            "carla_frame": sim_frame,
                            "timestamp_sec": round(timestamp_sec, 3),
                            "bbox": bbox,
                            "bbox_source": "carla_3d_projection",
                            "ocr_bbox": ocr_bbox,
                            "parking_slot_id": car.parking_slot_id if camera.camera_id.endswith("PARKING") else "",
                            "vehicle_actor_id": actor.id,
                            "camera_transform": transform_to_dict(camera_transform),
                            "world_transform": transform_to_dict(actor.get_transform()),
                        }
                        bbox_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    writers[camera.camera_id].write(frame)
    finally:
        for writer in writers.values():
            writer.release()
        for actor in actors_to_destroy:
            try:
                actor.destroy()
            except RuntimeError:
                pass
        world.apply_settings(original_settings)

    return carla_version


def validate_carla_connection(host: str, port: int) -> None:
    carla = import_carla_module()

    client = carla.Client(host, port)
    client.set_timeout(10.0)
    version = client.get_client_version()
    world = client.load_world("Town05")
    world.wait_for_tick()
    print(f"Connected to CARLA client {version}; loaded {world.get_map().name}.")


def draw_map_png(path: Path) -> None:
    width, height = 1220, 540
    img = Image.new("RGB", (width, height), (246, 248, 250))
    draw = ImageDraw.Draw(img)
    title_font = font(26)
    label_font = font(17)
    small_font = font(13)

    draw.text((34, 26), "Honda Smart Car Tracking POC - Camera Map", font=title_font, fill=(24, 32, 42))

    edges = [
        ("CAM_01_START_OCR", "CAM_02_TRANSIT"),
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

- CARLA target map: `Town05`
- Dataset scale: Small POC, 8-12 vehicles
- Cameras: 7 fixed virtual CCTV viewpoints rendered by CARLA RGB sensors
- Oil tank IDs: six-digit numeric strings only, starting at `100001`
- Status rule: at `CAM_03_JUNCTION_STATUS`, left turn is `GOOD` and right turn is `DEFECT`
- Parking slots: `G01-G06` for GOOD vehicles and `D01-D04` for DEFECT vehicles
- Bounding boxes: CARLA 3D vehicle bounding boxes projected into each camera plane
- OCR label: six-digit oil tank ID projected onto the windshield region in `CAM_01_START_OCR`

## Generated Files

- `{output_dir}/videos/*.mp4`
- `{output_dir}/metadata/cars.csv`
- `{output_dir}/metadata/camera_graph.json`
- `{output_dir}/metadata/events.jsonl`
- `{output_dir}/annotations/bboxes.jsonl`
- `docs/carla_honda_poc_map.png`

## Generate Locally

Run CARLA 0.9.15 first, then generate the real 3D CCTV dataset:

```bash
python scripts/generate_carla_dataset.py --renderer carla --clean
```

For metadata-only development without CARLA, use the storyboard dry-run:

```bash
python scripts/generate_carla_dataset.py --renderer storyboard --clean
```
"""
    path.write_text(text, encoding="utf-8")


def validate_outputs(output_dir: Path, cars: list[CarSpec]) -> None:
    oil_ids = [car.oil_tank_id for car in cars]
    if len(oil_ids) != len(set(oil_ids)):
        raise RuntimeError("oil_tank_id values must be unique.")
    for oil_id in oil_ids:
        if not re.fullmatch(r"\d{6}", oil_id):
            raise RuntimeError(f"Invalid oil_tank_id: {oil_id}")

    videos = sorted((output_dir / "videos").glob("*.mp4"))
    if len(videos) != len(CAMERAS):
        raise RuntimeError(f"Expected {len(CAMERAS)} videos, found {len(videos)}.")

    required = [
        output_dir / "metadata" / "cars.csv",
        output_dir / "metadata" / "camera_graph.json",
        output_dir / "metadata" / "events.jsonl",
        output_dir / "annotations" / "bboxes.jsonl",
    ]
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")

    with (output_dir / "metadata" / "camera_graph.json").open(encoding="utf-8") as fh:
        graph = json.load(fh)
    if len(graph.get("cameras", [])) != len(CAMERAS):
        raise RuntimeError(f"Expected {len(CAMERAS)} cameras in camera_graph.json.")


def main() -> int:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    docs_dir = Path(args.docs_dir)
    paths = ensure_dirs(output_dir, docs_dir, args.clean)
    cars = make_cars(args.num_cars, args.oil_start)

    if args.renderer == "carla":
        carla_version = render_carla(
            paths,
            cars,
            args.carla_host,
            args.carla_port,
            args.fps,
            args.duration_sec,
            args.width,
            args.height,
        )
        print(f"Generated CARLA 3D CCTV videos with CARLA {carla_version}.")
    else:
        write_camera_graph(paths["metadata"] / "camera_graph.json")
        write_cars_csv(paths["metadata"] / "cars.csv", cars)
        write_events(paths["metadata"] / "events.jsonl", cars)
        render_storyboard(paths, cars, args.fps, args.duration_sec, args.width, args.height)
        print("Generated storyboard dry-run videos. Use --renderer carla for the final 3D CCTV dataset.")

    draw_map_png(docs_dir / "carla_honda_poc_map.png")
    write_dataset_doc(docs_dir / "carla_honda_poc_dataset.md", output_dir)
    validate_outputs(output_dir, cars)

    print(f"Generated {len(CAMERAS)} videos for {len(cars)} cars in {output_dir}")
    print(f"Map written to {docs_dir / 'carla_honda_poc_map.png'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
