#!/usr/bin/env python3
"""Compose the Honda CCTV visual map with YOLO26 pixel-based tracking boxes.

This variant intentionally does not use CARLA 3D projection annotations for
vehicle boxes. Boxes come only from YOLO tracking on the actual camera videos,
which better simulates what a real YOLO pipeline would see.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    tqdm = None

import compose_monitor_room as monitor


CLASSIFICATION_CAMERA_ID = "CAM_03_JUNCTION_STATUS"
PRE_CLASSIFICATION_CAMERAS = {"CAM_01_START", "CAM_02_TRANSIT"}
GOOD_CAMERAS = {"CAM_04_GOOD_ROUTE", "CAM_06_GOOD_PARKING"}
DEFECT_CAMERAS = {"CAM_05_DEFECT_ROUTE", "CAM_07_DEFECT_PARKING"}

CAM3_BASE_SIZE = monitor.CAM3_BASE_SIZE
CAM3_GOOD_GATE = [(76, 166), (368, 150), (373, 209), (66, 209)]
CAM3_DEFECT_GATE = [(0, 38), (92, 50), (104, 158), (0, 192)]


@dataclass(frozen=True)
class YoloRecord:
    camera_id: str
    frame_id: int
    local_track_id: int
    bbox: list[float]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackSegment:
    camera_id: str
    local_track_id: int
    first_frame: int
    last_frame: int
    best_confidence: float
    global_id: int | None = None
    status: str | None = None


def resolve_yolo_device(requested_device: str) -> str | None:
    try:
        import torch
    except ModuleNotFoundError:
        return requested_device or None

    if requested_device:
        if requested_device.lower() == "cpu":
            return "cpu"
        try:
            device_idx = int(requested_device)
        except ValueError:
            return requested_device
        if not torch.cuda.is_available():
            print(f"[yolo] Requested CUDA device {requested_device}, but CUDA is unavailable. Falling back to CPU.")
            return "cpu"
        major, minor = torch.cuda.get_device_capability(device_idx)
        arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if arch not in supported_arches:
            name = torch.cuda.get_device_name(device_idx)
            print(
                f"[yolo] Requested CUDA device {requested_device} ({name} {arch}) is not supported by the "
                "installed PyTorch build. Falling back to CPU."
            )
            return "cpu"
        return requested_device

    if not torch.cuda.is_available():
        print("[yolo] CUDA is not available. Falling back to CPU.")
        return "cpu"

    supported_arches = set(torch.cuda.get_arch_list())
    for device_idx in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(device_idx)
        arch = f"sm_{major}{minor}"
        if arch in supported_arches:
            return str(device_idx)

    gpu_summaries = []
    for device_idx in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(device_idx)
        name = torch.cuda.get_device_name(device_idx)
        gpu_summaries.append(f"{device_idx}:{name} sm_{major}{minor}")
    print(
        "[yolo] Installed PyTorch was not built for this GPU architecture "
        f"({', '.join(gpu_summaries)}). Falling back to CPU."
    )
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose seven CCTV videos into a visual-map monitor using YOLO26 tracking boxes."
    )
    parser.add_argument("--input-dir", default="datasets/carla_honda_poc_aaa_relight/videos")
    parser.add_argument("--output", default="datasets/carla_honda_poc_aaa_relight/videos/visual_map_monitor_yolo26.mp4")
    parser.add_argument("--metadata-dir", default="datasets/carla_honda_poc_aaa_relight/metadata")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS. Default uses the first input video FPS.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap for quick previews.")
    parser.add_argument("--show-frame-id", action="store_true")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--yolo-model", default="yolo26m.pt", help="Ultralytics YOLO model, e.g. yolo26m.pt.")
    parser.add_argument("--yolo-conf", type=float, default=0.10,
                        help="Detection confidence threshold. Default 0.10 (lower suits CARLA synthetic footage).")
    parser.add_argument("--yolo-imgsz", type=int, default=1280,
                        help="Inference image size (long side). Default 1280 for better small-object recall.")
    parser.add_argument("--yolo-iou", type=float, default=0.45,
                        help="NMS IoU threshold. Default 0.45.")
    parser.add_argument("--yolo-augment", action="store_true",
                        help="Test-time augmentation (TTA). Increases recall at the cost of speed.")
    parser.add_argument("--yolo-half", action="store_true",
                        help="FP16 half-precision inference (GPU only). ~2x faster on supported hardware.")
    parser.add_argument("--yolo-device", default="", help="Ultralytics device string. Empty lets Ultralytics decide.")
    parser.add_argument("--yolo-tracker", default="bytetrack.yaml")
    parser.add_argument(
        "--yolo-source-mode",
        choices=("sequential", "streams"),
        default="sequential",
        help="YOLO cache build mode. 'streams' runs all camera videos together as multi-stream inference.",
    )
    parser.add_argument(
        "--vehicle-class-ids",
        default="2,5,7",
        help="Comma-separated COCO class IDs to track. Default: car,bus,truck.",
    )
    parser.add_argument(
        "--yolo-cache",
        default="",
        help="JSONL cache path. Default: <metadata-dir>/yolo26_tracks.jsonl.",
    )
    parser.add_argument("--rebuild-yolo-cache", action="store_true")
    parser.add_argument(
        "--preprocess-clahe",
        action="store_true",
        help="Apply CLAHE contrast normalisation to each frame before inference. Helps with CARLA synthetic lighting.",
    )
    parser.add_argument(
        "--debug-frames",
        type=int,
        default=0,
        help="If >0, extract this many evenly-spaced frames per camera and save annotated images to "
             "<metadata-dir>/debug_yolo/ for tuning. Uses conf=0.01 to show all candidates.",
    )
    return parser.parse_args()


def parse_class_ids(value: str) -> list[int]:
    class_ids: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        class_ids.append(int(item))
    return class_ids


def default_yolo_cache(metadata_dir: Path, source_mode: str = "sequential") -> Path:
    suffix = "" if source_mode == "sequential" else f"_{source_mode}"
    return metadata_dir / f"yolo26_tracks{suffix}.jsonl"


def log(message: str) -> None:
    print(message, flush=True)


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """Normalise contrast via CLAHE on the L-channel (LAB). Reduces CARLA synthetic-lighting artefacts."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _clahe_frame_generator(video_path: Path, max_frames: int):
    """Yield CLAHE-processed BGR frames from *video_path* for use as a YOLO source."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        frame_idx = 0
        while True:
            if max_frames > 0 and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield apply_clahe(frame)
            frame_idx += 1
    finally:
        cap.release()


class ManualProgress:
    def __init__(self, iterable, total: int | None = None, desc: str = "", unit: str = "it") -> None:
        self.iterable = iterable
        self.total = total
        self.desc = desc or "Progress"
        self.unit = unit
        self.current = 0
        self.postfix = ""
        self.last_report = 0.0

    def set_postfix_str(self, postfix: str) -> None:
        self.postfix = postfix

    def _report(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_report < 1.0:
            return
        self.last_report = now
        if self.total:
            pct = self.current / self.total * 100.0
            suffix = f" | {self.postfix}" if self.postfix else ""
            log(f"[progress] {self.desc}: {self.current}/{self.total} {self.unit} ({pct:5.1f}%){suffix}")
        else:
            suffix = f" | {self.postfix}" if self.postfix else ""
            log(f"[progress] {self.desc}: {self.current} {self.unit}{suffix}")

    def __iter__(self):
        for item in self.iterable:
            self.current += 1
            self._report()
            yield item
        self._report(force=True)


def progress(iterable, total: int | None = None, desc: str = "", unit: str = "it"):
    if tqdm is None:
        return ManualProgress(iterable, total=total, desc=desc, unit=unit)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        ascii=True,
        mininterval=0.2,
        smoothing=0.05,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    try:
        return max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()


def camera_video_paths(input_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for camera_id in monitor.CAMERA_IDS:
        video_path = input_dir / f"{camera_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing camera video: {video_path}")
        paths[camera_id] = video_path
    return paths


def infer_camera_id_from_result_path(result_path: str) -> str:
    for camera_id in monitor.CAMERA_IDS:
        if camera_id in result_path:
            return camera_id
    raise KeyError(f"Could not map YOLO stream result path to camera ID: {result_path}")


def cache_summary_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".summary.json")


def write_cache_summary(cache_path: Path, source_mode: str, frame_counts: dict[str, int], total_records: int) -> None:
    summary = {
        "source_mode": source_mode,
        "frame_counts": frame_counts,
        "total_records": total_records,
        "cameras": list(monitor.CAMERA_IDS),
        "complete": all(frame_counts.get(camera_id, 0) > 0 for camera_id in monitor.CAMERA_IDS),
    }
    cache_summary_path(cache_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def maybe_warn_on_cache_summary(cache_path: Path) -> None:
    summary_path = cache_summary_path(cache_path)
    if not summary_path.exists():
        log(
            f"[monitor-yolo] Warning: cache summary is missing for {cache_path.name}. "
            "If this cache came from an interrupted run, rebuild it with --rebuild-yolo-cache."
        )
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log(
            f"[monitor-yolo] Warning: cache summary is unreadable for {cache_path.name}. "
            "Consider rebuilding with --rebuild-yolo-cache."
        )
        return
    frame_counts = summary.get("frame_counts", {})
    missing = [camera_id for camera_id in monitor.CAMERA_IDS if int(frame_counts.get(camera_id, 0) or 0) <= 0]
    if missing:
        log(
            f"[monitor-yolo] Warning: cache summary shows no processed frames for {', '.join(missing)}. "
            "This usually means the cache build stopped early. Rebuild with --rebuild-yolo-cache."
        )


def read_yolo_index(path: Path) -> dict[tuple[str, int], list[YoloRecord]]:
    index: defaultdict[tuple[str, int], list[YoloRecord]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                item = YoloRecord(
                    camera_id=str(record["camera_id"]),
                    frame_id=int(record["frame_id"]),
                    local_track_id=int(record["local_track_id"]),
                    bbox=[float(v) for v in record["bbox"]],
                    confidence=float(record.get("confidence", 0.0)),
                    class_id=int(record.get("class_id", -1)),
                    class_name=str(record.get("class_name", "")),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid YOLO cache record at {path}:{line_no}") from exc
            if item.camera_id in monitor.CAMERA_IDS and len(item.bbox) == 4:
                index[(item.camera_id, item.frame_id)].append(item)
    return dict(index)


def write_yolo_cache_sequential(
    input_dir: Path,
    output_path: Path,
    model_name: str,
    tracker: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    class_ids: list[int],
    max_frames: int,
    augment: bool = False,
    half: bool = False,
    preprocess_clahe: bool = False,
) -> None:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO26 tracking. Install it with: pip install -r requirements.txt"
        ) from exc

    resolved_device = resolve_yolo_device(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    total_records = 0
    frame_counts: dict[str, int] = {}
    log(
        "[yolo] Starting cache build: "
        f"model={model_name} device={resolved_device or 'auto'} tracker={tracker} imgsz={imgsz} "
        f"conf={conf} iou={iou} augment={augment} half={half} clahe={preprocess_clahe}"
    )
    with output_path.open("w", encoding="utf-8") as fh:
        for camera_id, video_path in camera_video_paths(input_dir).items():
            log(f"[yolo] Preparing {camera_id} from {video_path.name}")
            total_frames = video_frame_count(video_path)
            if max_frames > 0 and total_frames > 0:
                total_frames = min(total_frames, max_frames)
            log(
                f"[yolo] {camera_id}: total_frames={total_frames or 'unknown'} "
                f"class_ids={class_ids or 'all'} warmup=starting"
            )
            source: str | list[np.ndarray]
            if preprocess_clahe:
                source = _clahe_frame_generator(video_path, max_frames)
            else:
                source = str(video_path)
            stream = model.track(
                source=source,
                stream=True,
                tracker=tracker,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=resolved_device,
                classes=class_ids or None,
                augment=augment,
                half=half,
                persist=False,
                verbose=False,
            )
            stream_iter = progress(
                enumerate(stream),
                total=total_frames if total_frames > 0 else None,
                desc=f"YOLO {camera_id}",
                unit="frame",
            )
            camera_records_before = total_records
            last_frame_idx = -1
            for frame_idx, result in stream_iter:
                if max_frames > 0 and frame_idx >= max_frames:
                    break
                last_frame_idx = frame_idx
                boxes = getattr(result, "boxes", None)
                if boxes is None or boxes.id is None:
                    if tqdm is not None:
                        stream_iter.set_postfix_str(f"boxes={total_records}")
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(ids), dtype=np.float32)
                classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.full(len(ids), -1, dtype=np.float32)
                names = getattr(result, "names", {}) or {}
                for bbox, track_id, det_conf, class_id in zip(xyxy, ids, confs, classes):
                    class_id_int = int(class_id)
                    payload = {
                        "camera_id": camera_id,
                        "frame_id": int(frame_idx),
                        "local_track_id": int(track_id),
                        "bbox": [float(v) for v in bbox.tolist()],
                        "confidence": float(det_conf),
                        "class_id": class_id_int,
                        "class_name": str(names.get(class_id_int, class_id_int)),
                    }
                    fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
                    total_records += 1
                if tqdm is not None:
                    stream_iter.set_postfix_str(f"tracks={len(ids)} cache={total_records}")
            log(
                f"[yolo] Completed {camera_id}: "
                f"new_records={total_records - camera_records_before} total_cache_records={total_records}"
            )
            frame_counts[camera_id] = last_frame_idx + 1
    write_cache_summary(output_path, "sequential", frame_counts, total_records)
    log(f"[yolo] Wrote cache: {output_path} ({total_records} boxes)")


def write_yolo_cache_streams(
    input_dir: Path,
    output_path: Path,
    model_name: str,
    tracker: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    class_ids: list[int],
    max_frames: int,
    augment: bool = False,
    half: bool = False,
) -> None:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO26 tracking. Install it with: pip install -r requirements.txt"
        ) from exc

    resolved_device = resolve_yolo_device(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    video_paths = camera_video_paths(input_dir)
    total_frames_by_camera = {camera_id: video_frame_count(path) for camera_id, path in video_paths.items()}
    total_batches = min(frame_count for frame_count in total_frames_by_camera.values() if frame_count > 0)
    if max_frames > 0:
        total_batches = min(total_batches, max_frames)
    total_results = total_batches * len(monitor.CAMERA_IDS) if total_batches > 0 else None

    log(
        "[yolo] Starting cache build: "
        f"model={model_name} device={resolved_device or 'auto'} tracker={tracker} imgsz={imgsz} "
        f"conf={conf} iou={iou} augment={augment} half={half} source_mode=streams"
    )
    for camera_id, frame_count in total_frames_by_camera.items():
        log(f"[yolo] Stream source {camera_id}: total_frames={frame_count}")

    stream_list_file: str | None = None
    total_records = 0
    frame_counts = {camera_id: 0 for camera_id in monitor.CAMERA_IDS}
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".streams", delete=False, encoding="utf-8") as tmp:
            tmp.write("\n".join(str(video_paths[camera_id]) for camera_id in monitor.CAMERA_IDS))
            tmp.write("\n")
            stream_list_file = tmp.name

        stream = model.track(
            source=stream_list_file,
            stream=True,
            tracker=tracker,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=resolved_device,
            classes=class_ids or None,
            augment=augment,
            half=half,
            persist=False,
            verbose=False,
            batch=len(monitor.CAMERA_IDS),
        )

        with output_path.open("w", encoding="utf-8") as fh:
            stream_iter = progress(
                stream,
                total=total_results,
                desc="YOLO 7-camera streams",
                unit="result",
            )
            for result in stream_iter:
                camera_id = infer_camera_id_from_result_path(str(result.path))
                local_frame_idx = frame_counts[camera_id]
                frame_counts[camera_id] += 1

                boxes = getattr(result, "boxes", None)
                if boxes is not None and boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    ids = boxes.id.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(ids), dtype=np.float32)
                    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.full(len(ids), -1, dtype=np.float32)
                    names = getattr(result, "names", {}) or {}
                    for bbox, track_id, det_conf, class_id in zip(xyxy, ids, confs, classes):
                        class_id_int = int(class_id)
                        payload = {
                            "camera_id": camera_id,
                            "frame_id": int(local_frame_idx),
                            "local_track_id": int(track_id),
                            "bbox": [float(v) for v in bbox.tolist()],
                            "confidence": float(det_conf),
                            "class_id": class_id_int,
                            "class_name": str(names.get(class_id_int, class_id_int)),
                        }
                        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
                        total_records += 1

                if tqdm is not None:
                    active_tracks = 0 if boxes is None or boxes.id is None else len(boxes.id)
                    stream_iter.set_postfix_str(
                        f"cam={camera_id} frame={local_frame_idx + 1}/{total_batches} tracks={active_tracks} cache={total_records}"
                    )

                if total_batches > 0 and min(frame_counts.values()) >= total_batches:
                    break
    finally:
        if stream_list_file:
            Path(stream_list_file).unlink(missing_ok=True)

    for camera_id in monitor.CAMERA_IDS:
        log(f"[yolo] Completed {camera_id}: processed_frames={frame_counts[camera_id]}")
    write_cache_summary(output_path, "streams", frame_counts, total_records)
    log(f"[yolo] Wrote cache: {output_path} ({total_records} boxes)")


def write_yolo_cache(
    input_dir: Path,
    output_path: Path,
    model_name: str,
    tracker: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    class_ids: list[int],
    max_frames: int,
    source_mode: str,
    augment: bool = False,
    half: bool = False,
    preprocess_clahe: bool = False,
) -> None:
    if source_mode == "streams":
        write_yolo_cache_streams(
            input_dir=input_dir,
            output_path=output_path,
            model_name=model_name,
            tracker=tracker,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            class_ids=class_ids,
            max_frames=max_frames,
            augment=augment,
            half=half,
        )
        return
    write_yolo_cache_sequential(
        input_dir=input_dir,
        output_path=output_path,
        model_name=model_name,
        tracker=tracker,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        class_ids=class_ids,
        max_frames=max_frames,
        augment=augment,
        half=half,
        preprocess_clahe=preprocess_clahe,
    )


def build_or_load_yolo_index(args: argparse.Namespace, input_dir: Path, metadata_dir: Path) -> dict[tuple[str, int], list[YoloRecord]]:
    cache_path = Path(args.yolo_cache) if args.yolo_cache else default_yolo_cache(metadata_dir, args.yolo_source_mode)
    if args.max_frames > 0 and not args.yolo_cache:
        cache_path = metadata_dir / f"yolo26_tracks_{args.yolo_source_mode}_preview_{args.max_frames}.jsonl"
    if args.rebuild_yolo_cache or not cache_path.exists():
        log(f"[monitor-yolo] Building YOLO cache at {cache_path}")
        write_yolo_cache(
            input_dir=input_dir,
            output_path=cache_path,
            model_name=args.yolo_model,
            tracker=args.yolo_tracker,
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            imgsz=args.yolo_imgsz,
            device=args.yolo_device,
            class_ids=parse_class_ids(args.vehicle_class_ids),
            max_frames=args.max_frames,
            source_mode=args.yolo_source_mode,
            augment=args.yolo_augment,
            half=args.yolo_half,
            preprocess_clahe=args.preprocess_clahe,
        )
    else:
        log(f"[monitor-yolo] Reusing existing YOLO cache: {cache_path}")
    maybe_warn_on_cache_summary(cache_path)
    index = read_yolo_index(cache_path)
    log(f"[monitor-yolo] Loaded YOLO cache: {cache_path} ({sum(len(v) for v in index.values())} boxes)")
    return index


def segments_from_index(index: dict[tuple[str, int], list[YoloRecord]]) -> dict[tuple[str, int], TrackSegment]:
    segments: dict[tuple[str, int], TrackSegment] = {}
    for (camera_id, frame_id), records in index.items():
        for record in records:
            key = (camera_id, record.local_track_id)
            segment = segments.get(key)
            if segment is None:
                segments[key] = TrackSegment(
                    camera_id=camera_id,
                    local_track_id=record.local_track_id,
                    first_frame=frame_id,
                    last_frame=frame_id,
                    best_confidence=record.confidence,
                )
            else:
                segment.first_frame = min(segment.first_frame, frame_id)
                segment.last_frame = max(segment.last_frame, frame_id)
                segment.best_confidence = max(segment.best_confidence, record.confidence)
    return segments


def records_for_segment(
    index: dict[tuple[str, int], list[YoloRecord]],
    camera_id: str,
    local_track_id: int,
) -> list[YoloRecord]:
    records: list[YoloRecord] = []
    for (record_camera_id, _frame_id), frame_records in index.items():
        if record_camera_id != camera_id:
            continue
        for record in frame_records:
            if record.local_track_id == local_track_id:
                records.append(record)
    records.sort(key=lambda item: item.frame_id)
    return records


def scaled_cam3_points(points: list[tuple[int, int]], target_w: int, target_h: int) -> list[tuple[int, int]]:
    base_w, base_h = CAM3_BASE_SIZE
    return [(int(round(x / base_w * target_w)), int(round(y / base_h * target_h))) for x, y in points]


def infer_cam3_segment_status(
    index: dict[tuple[str, int], list[YoloRecord]],
    segment: TrackSegment,
    src_sizes: dict[str, tuple[int, int]],
) -> str | None:
    src_w, src_h = src_sizes.get(segment.camera_id, (1280, 720))
    tile = monitor.LAYOUT_1080P[segment.camera_id]
    good_gate = scaled_cam3_points(CAM3_GOOD_GATE, tile.w, tile.h)
    defect_gate = scaled_cam3_points(CAM3_DEFECT_GATE, tile.w, tile.h)
    good_hits = 0
    defect_hits = 0
    for record in records_for_segment(index, segment.camera_id, segment.local_track_id):
        mapped = monitor.map_bbox_to_tile(record.bbox, src_w, src_h, tile.w, tile.h)
        if mapped is None:
            continue
        x1, y1, x2, y2 = mapped
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if monitor.point_in_polygon(center, good_gate):
            good_hits += 1
        if monitor.point_in_polygon(center, defect_gate):
            defect_hits += 1
    if good_hits == 0 and defect_hits == 0:
        return None
    return "GOOD" if good_hits >= defect_hits else "DEFECT"


def assign_by_order(segments: list[TrackSegment], global_ids: list[int], status: str | None = None) -> None:
    for segment, global_id in zip(sorted(segments, key=lambda item: (item.first_frame, -item.best_confidence)), global_ids):
        segment.global_id = global_id
        if status is not None:
            segment.status = status


def build_track_assignments(
    index: dict[tuple[str, int], list[YoloRecord]],
    src_sizes: dict[str, tuple[int, int]],
) -> dict[tuple[str, int], TrackSegment]:
    segments = segments_from_index(index)
    by_camera: dict[str, list[TrackSegment]] = defaultdict(list)
    for segment in segments.values():
        if segment.last_frame - segment.first_frame >= 3:
            by_camera[segment.camera_id].append(segment)

    cam1 = sorted(by_camera.get("CAM_01_START", []), key=lambda item: (item.first_frame, -item.best_confidence))
    next_global_id = 1
    primary_ids: list[int] = []
    for segment in cam1:
        segment.global_id = next_global_id
        primary_ids.append(next_global_id)
        next_global_id += 1

    for camera_id in ("CAM_02_TRANSIT", "CAM_03_JUNCTION_STATUS"):
        assign_by_order(by_camera.get(camera_id, []), primary_ids)

    good_ids: list[int] = []
    defect_ids: list[int] = []
    for segment in by_camera.get(CLASSIFICATION_CAMERA_ID, []):
        status = infer_cam3_segment_status(index, segment, src_sizes)
        if status is None and segment.global_id is not None:
            status = "GOOD" if segment.global_id % 2 == 1 else "DEFECT"
        segment.status = status
        if segment.global_id is None:
            segment.global_id = next_global_id
            next_global_id += 1
        if status == "GOOD":
            good_ids.append(segment.global_id)
        elif status == "DEFECT":
            defect_ids.append(segment.global_id)

    if not good_ids and primary_ids:
        good_ids = primary_ids[::2]
    if not defect_ids and primary_ids:
        defect_ids = primary_ids[1::2]

    for camera_id in ("CAM_04_GOOD_ROUTE", "CAM_06_GOOD_PARKING"):
        assign_by_order(by_camera.get(camera_id, []), good_ids, "GOOD")
    for camera_id in ("CAM_05_DEFECT_ROUTE", "CAM_07_DEFECT_PARKING"):
        assign_by_order(by_camera.get(camera_id, []), defect_ids, "DEFECT")

    for segment in segments.values():
        if segment.global_id is None:
            segment.global_id = next_global_id
            next_global_id += 1
        if segment.status is None:
            if segment.camera_id in GOOD_CAMERAS:
                segment.status = "GOOD"
            elif segment.camera_id in DEFECT_CAMERAS:
                segment.status = "DEFECT"
    return segments


def status_for_record(
    camera_id: str,
    bbox: tuple[int, int, int, int],
    segment: TrackSegment | None,
    target_w: int,
    target_h: int,
) -> str | None:
    if segment is None:
        return None
    if camera_id in PRE_CLASSIFICATION_CAMERAS:
        return None
    if camera_id in GOOD_CAMERAS | DEFECT_CAMERAS:
        return segment.status
    if camera_id != CLASSIFICATION_CAMERA_ID:
        return None
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if monitor.point_in_polygon(center, scaled_cam3_points(CAM3_GOOD_GATE, target_w, target_h)):
        return "GOOD"
    if monitor.point_in_polygon(center, scaled_cam3_points(CAM3_DEFECT_GATE, target_w, target_h)):
        return "DEFECT"
    return None


def tracking_label(segment: TrackSegment | None, revealed_status: str | None) -> str:
    global_id = segment.global_id if segment is not None and segment.global_id is not None else 0
    if revealed_status in {"GOOD", "DEFECT"}:
        return f"ID {global_id} {revealed_status}"
    return f"ID {global_id}" if global_id else "ID ?"


def draw_cam3_yolo_decision_zones(image: np.ndarray) -> None:
    target_h, target_w = image.shape[:2]
    zones = [
        ("GOOD EXIT", scaled_cam3_points(CAM3_GOOD_GATE, target_w, target_h), monitor.ROUTE_COLORS["good"]),
        ("BAD EXIT", scaled_cam3_points(CAM3_DEFECT_GATE, target_w, target_h), monitor.ROUTE_COLORS["defect"]),
    ]
    for label, points, color in zones:
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.16, image, 0.84, 0.0, image)
        cv2.polylines(image, [np.array(points, dtype=np.int32)], True, color, 2, cv2.LINE_AA)
        cv2.polylines(image, [np.array(points, dtype=np.int32)], True, (245, 250, 248), 1, cv2.LINE_AA)

        min_x = min(p[0] for p in points)
        max_y = max(p[1] for p in points)
        text_x = max(5, min(target_w - 112, min_x + 7))
        text_y = max(18, min(target_h - 8, max_y - 9))
        tw, th = monitor.text_size(label, 0.28, 1)
        monitor.rounded_rect(image, (text_x - 4, text_y - th - 6), (text_x + tw + 6, text_y + 4), (8, 12, 14), 4)
        monitor.rounded_rect(image, (text_x - 4, text_y - th - 6), (text_x + tw + 6, text_y + 4), color, 4, 1)
        monitor.put_text(image, label, (text_x, text_y - 2), 0.28, (235, 240, 240), 1)


def draw_yolo_boxes(
    image: np.ndarray,
    camera_id: str,
    records: list[YoloRecord],
    src_shape: tuple[int, int],
    segments: dict[tuple[str, int], TrackSegment],
) -> None:
    src_h, src_w = src_shape
    target_h, target_w = image.shape[:2]
    for record in records:
        mapped = monitor.map_bbox_to_tile(record.bbox, src_w, src_h, target_w, target_h)
        if mapped is None:
            continue
        x1, y1, x2, y2 = mapped
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        segment = segments.get((camera_id, record.local_track_id))
        revealed_status = status_for_record(camera_id, mapped, segment, target_w, target_h)
        color = monitor.bbox_color(revealed_status)

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

        label = tracking_label(segment, revealed_status)
        tw, th = monitor.text_size(label, 0.32, 1)
        label_x = max(0, min(target_w - tw - 10, x1))
        label_y = y1 - 7 if y1 >= th + 43 else min(target_h - 4, y2 + th + 10)
        box1 = (label_x, max(0, label_y - th - 8))
        box2 = (label_x + tw + 10, min(target_h - 1, label_y + 4))
        monitor.rounded_rect(image, box1, box2, (8, 12, 14), 4)
        monitor.rounded_rect(image, box1, box2, color, 4, 1)
        monitor.put_text(image, label, (label_x + 5, label_y - 2), 0.32, (235, 240, 240), 1)


def draw_tile_yolo(
    canvas: np.ndarray,
    spec: monitor.TileSpec,
    frame: np.ndarray,
    frame_idx: int,
    yolo_records: list[YoloRecord],
    segments: dict[tuple[str, int], TrackSegment],
) -> None:
    color = monitor.CAMERA_COLORS[spec.camera_id]
    shadow = np.zeros_like(canvas)
    monitor.rounded_rect(shadow, (spec.x - 8, spec.y - 8), (spec.x + spec.w + 8, spec.y + spec.h + 34), (0, 0, 0), 8)
    cv2.addWeighted(shadow, 0.34, canvas, 1.0, 0.0, canvas)

    monitor.rounded_rect(canvas, (spec.x - 3, spec.y - 3), (spec.x + spec.w + 3, spec.y + spec.h + 30), (10, 14, 17), 7)
    monitor.rounded_rect(canvas, (spec.x - 3, spec.y - 3), (spec.x + spec.w + 3, spec.y + spec.h + 30), (112, 128, 132), 7, 1)
    cv2.line(canvas, (spec.x, spec.y - 3), (spec.x + spec.w, spec.y - 3), color, 2, cv2.LINE_AA)

    image = monitor.cover_resize(frame, spec.w, spec.h)
    image = cv2.addWeighted(image, 0.92, np.zeros_like(image), 0.08, 0.0)
    if spec.camera_id == CLASSIFICATION_CAMERA_ID:
        draw_cam3_yolo_decision_zones(image)
    if yolo_records:
        draw_yolo_boxes(image, spec.camera_id, yolo_records, frame.shape[:2], segments)

    canvas[spec.y : spec.y + spec.h, spec.x : spec.x + spec.w] = image
    cv2.rectangle(canvas, (spec.x, spec.y), (spec.x + spec.w, spec.y + spec.h), (90, 105, 108), 1, cv2.LINE_AA)

    header_h = 30
    overlay = canvas.copy()
    cv2.rectangle(overlay, (spec.x, spec.y), (spec.x + spec.w, spec.y + header_h), (8, 10, 11), -1)
    cv2.addWeighted(overlay, 0.64, canvas, 0.36, 0.0, canvas)
    short_name = monitor.CAMERA_NAMES[spec.camera_id].replace("CAM ", "C").replace("  ", "_", 1)
    monitor.put_text(canvas, short_name, (spec.x + 12, spec.y + 20), 0.42, color, 2)
    monitor.put_text(canvas, "YOLO26", (spec.x + spec.w - 78, spec.y + 20), 0.31, (210, 220, 220), 1)
    cv2.circle(canvas, (spec.x + spec.w - 14, spec.y + 15), 4, (45, 65, 255), -1, cv2.LINE_AA)

    footer_y = spec.y + spec.h + 21
    monitor.put_text(canvas, monitor.CAMERA_SUBTITLES[spec.camera_id], (spec.x + 10, footer_y), 0.35, (172, 184, 184), 1)
    monitor.put_text(canvas, f"{frame_idx:05d}", (spec.x + spec.w - 62, footer_y), 0.33, (118, 130, 130), 1)


def compose_frame(
    frames: dict[str, np.ndarray],
    yolo_index: dict[tuple[str, int], list[YoloRecord]],
    segments: dict[tuple[str, int], TrackSegment],
    width: int,
    height: int,
    frame_idx: int,
    fps: float,
    metadata: dict[str, object],
    show_frame_id: bool,
) -> np.ndarray:
    canvas = monitor.draw_background(width, height)
    monitor.draw_header(canvas, frame_idx, fps, metadata, show_frame_id)
    monitor.draw_flow(canvas, frame_idx)
    for camera_id in monitor.CAMERA_IDS:
        draw_tile_yolo(
            canvas,
            monitor.LAYOUT_1080P[camera_id],
            frames[camera_id],
            frame_idx,
            yolo_index.get((camera_id, frame_idx), []),
            segments,
        )
    monitor.draw_footer(canvas)
    return canvas


def capture_source_sizes(captures: dict[str, cv2.VideoCapture]) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    for camera_id, cap in captures.items():
        sizes[camera_id] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return sizes


def debug_detection_frames(
    input_dir: Path,
    debug_dir: Path,
    model_name: str,
    imgsz: int,
    device: str,
    class_ids: list[int],
    n_frames: int,
    preprocess_clahe: bool = False,
) -> None:
    """Extract *n_frames* evenly-spaced frames per camera and save YOLO-annotated images at conf=0.01.

    Images are written to *debug_dir* with filenames like ``CAM_01_START_frame00042.jpg``.
    Using conf=0.01 shows all candidates so you can tune --yolo-conf without a full cache rebuild.
    """
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ultralytics is required for debug frame extraction. Install it with: pip install -r requirements.txt"
        ) from exc

    resolved_device = resolve_yolo_device(device)
    debug_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    log(f"[debug] Writing {n_frames} annotated frames per camera to {debug_dir}")
    for camera_id, video_path in camera_video_paths(input_dir).items():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log(f"[debug] Cannot open {video_path}, skipping.")
            continue
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        step = max(1, total // n_frames)
        saved = 0
        for i in range(n_frames):
            pos = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                break
            if preprocess_clahe:
                frame = apply_clahe(frame)
            results = model.predict(
                frame,
                conf=0.01,
                iou=0.45,
                imgsz=imgsz,
                device=resolved_device,
                classes=class_ids or None,
                verbose=False,
            )
            annotated = results[0].plot()
            n_det = len(results[0].boxes) if results[0].boxes is not None else 0
            out_path = debug_dir / f"{camera_id}_frame{pos:05d}.jpg"
            cv2.imwrite(str(out_path), annotated)
            log(f"[debug] {out_path.name}: {n_det} detections at conf>=0.01")
            saved += 1
        cap.release()
        log(f"[debug] {camera_id}: saved {saved} debug frames")


def main() -> int:
    args = parse_args()
    monitor.require_1080p_layout(args.width, args.height)

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    metadata_dir = Path(args.metadata_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.debug_frames > 0:
        debug_detection_frames(
            input_dir=input_dir,
            debug_dir=metadata_dir / "debug_yolo",
            model_name=args.yolo_model,
            imgsz=args.yolo_imgsz,
            device=args.yolo_device,
            class_ids=parse_class_ids(args.vehicle_class_ids),
            n_frames=args.debug_frames,
            preprocess_clahe=args.preprocess_clahe,
        )
        return 0

    captures = monitor.open_captures(input_dir)
    src_sizes = capture_source_sizes(captures)
    metadata = monitor.read_camera_graph(metadata_dir)
    yolo_index = build_or_load_yolo_index(args, input_dir, metadata_dir)
    segments = build_track_assignments(yolo_index, src_sizes)

    first_fps, frame_count = monitor.input_video_info(captures)
    output_fps = args.fps if args.fps > 0 else first_fps
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)

    fourcc = cv2.VideoWriter_fourcc(*args.codec[:4])
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {output_path}")

    log(
        "[monitor-yolo] Starting compose: "
        f"frames={frame_count} fps={output_fps:.2f} output={output_path.name}"
    )
    try:
        frame_iter = progress(range(frame_count), total=frame_count, desc="Compose monitor", unit="frame")
        for frame_idx in frame_iter:
            frames: dict[str, np.ndarray] = {}
            for camera_id, cap in captures.items():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Could not read frame {frame_idx} from {camera_id}.")
                frames[camera_id] = frame
            canvas = compose_frame(
                frames,
                yolo_index,
                segments,
                args.width,
                args.height,
                frame_idx,
                output_fps,
                metadata,
                args.show_frame_id,
            )
            writer.write(canvas)
            if tqdm is not None:
                frame_iter.set_postfix_str(f"frame={frame_idx + 1} out={output_path.name}")
            elif frame_idx == 0 or (frame_idx + 1) % max(1, int(output_fps) * 5) == 0 or frame_idx + 1 == frame_count:
                pct = (frame_idx + 1) / frame_count * 100.0
                log(f"[monitor-yolo] frame {frame_idx + 1}/{frame_count} ({pct:5.1f}%)")
    finally:
        writer.release()
        for cap in captures.values():
            cap.release()

    log(f"[monitor-yolo] Wrote YOLO monitor-room video: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
