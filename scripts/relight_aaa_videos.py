#!/usr/bin/env python3
"""Relight already-rendered AAA CCTV videos without rerunning CARLA.

The pass is designed for the Honda CARLA AAA outputs where shadows can be a bit
too dense. It lifts shadows and midtones with a filmic curve while preserving
highlights, asphalt contrast, and the CCTV mood.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relight AAA CCTV videos by lifting shadows naturally.")
    parser.add_argument("--input-dir", default="datasets/carla_honda_poc_aaa")
    parser.add_argument("--output-dir", default="datasets/carla_honda_poc_aaa_relight")
    parser.add_argument("--video-subdir", default="videos")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--preset", choices=("subtle", "balanced", "stronger"), default="balanced")
    parser.add_argument("--shadow-lift", type=float, default=None, help="0.0-0.6 strength for dark-region lift.")
    parser.add_argument("--midtone-lift", type=float, default=None, help="0.0-0.4 strength for midtone lift.")
    parser.add_argument("--highlight-protect", type=float, default=None, help="0.0-1.0 highlight preservation strength.")
    parser.add_argument("--warmth", type=float, default=None, help="Small positive value warms sunlight/asphalt tones.")
    parser.add_argument("--saturation", type=float, default=None)
    parser.add_argument("--local-contrast", type=float, default=None)
    parser.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap for quick previews.")
    parser.add_argument("--copy-sidecars", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("shadow_lift", "midtone_lift", "highlight_protect"):
        value = getattr(args, name)
        if value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.saturation <= 0.0:
        raise ValueError("--saturation must be positive.")
    if args.local_contrast < 0.0:
        raise ValueError("--local-contrast must be non-negative.")


def apply_preset_defaults(args: argparse.Namespace) -> None:
    presets = {
        "subtle": {
            "shadow_lift": 0.090,
            "midtone_lift": 0.045,
            "highlight_protect": 0.86,
            "warmth": 0.010,
            "saturation": 1.020,
            "local_contrast": 0.24,
        },
        "balanced": {
            "shadow_lift": 0.155,
            "midtone_lift": 0.075,
            "highlight_protect": 0.82,
            "warmth": 0.014,
            "saturation": 1.025,
            "local_contrast": 0.28,
        },
        "stronger": {
            "shadow_lift": 0.215,
            "midtone_lift": 0.095,
            "highlight_protect": 0.78,
            "warmth": 0.018,
            "saturation": 1.030,
            "local_contrast": 0.30,
        },
    }
    defaults = presets[args.preset]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def list_videos(video_dir: Path) -> list[Path]:
    videos = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}")
    return videos


def copy_sidecars(input_dir: Path, output_dir: Path, video_subdir: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in input_dir.iterdir():
        if child.name == video_subdir:
            continue
        target = output_dir / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def write_profile(output_dir: Path, args: argparse.Namespace, videos: list[Path]) -> None:
    profile = {
        "profile": "aaa_relight",
        "input_dir": str(Path(args.input_dir)),
        "output_dir": str(Path(args.output_dir)),
        "video_count": len(videos),
        "settings": {
            "shadow_lift": args.shadow_lift,
            "midtone_lift": args.midtone_lift,
            "highlight_protect": args.highlight_protect,
            "warmth": args.warmth,
            "saturation": args.saturation,
            "local_contrast": args.local_contrast,
            "denoise": args.denoise,
        },
    }
    (output_dir / "relight_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")


def luminance_weights(frame_float: np.ndarray) -> np.ndarray:
    b = frame_float[:, :, 0]
    g = frame_float[:, :, 1]
    r = frame_float[:, :, 2]
    return 0.0722 * b + 0.7152 * g + 0.2126 * r


def apply_relight(
    frame: np.ndarray,
    shadow_lift: float,
    midtone_lift: float,
    highlight_protect: float,
    warmth: float,
    saturation: float,
    local_contrast: float,
    denoise: bool,
) -> np.ndarray:
    source = frame.astype(np.float32) / 255.0
    luminance = luminance_weights(source)

    shadow_mask = np.clip((0.52 - luminance) / 0.52, 0.0, 1.0) ** 1.85
    midtone_mask = np.clip(1.0 - np.abs(luminance - 0.47) / 0.47, 0.0, 1.0) ** 1.2
    highlight_mask = np.clip((luminance - 0.70) / 0.30, 0.0, 1.0)
    protect = 1.0 - highlight_mask * highlight_protect

    lifted = source.copy()
    lift = shadow_lift * shadow_mask + midtone_lift * midtone_mask
    lifted += lift[:, :, None] * protect[:, :, None]

    # Keep blacks cinematic instead of washing them out.
    black_guard = np.clip((luminance - 0.035) / 0.16, 0.0, 1.0)
    lifted = source + (lifted - source) * black_guard[:, :, None]
    lifted = np.maximum(lifted, source * (0.97 + 0.03 * protect[:, :, None]))
    lifted = np.clip(lifted, 0.0, 1.0)

    lab = cv2.cvtColor((lifted * 255.0).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0 + local_contrast, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lifted = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(np.clip(lifted * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    lifted = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    lifted[:, :, 2] += warmth * (shadow_mask * 0.35 + midtone_mask * 0.65)
    lifted[:, :, 0] -= warmth * 0.45 * midtone_mask
    lifted = np.clip(lifted, 0.0, 1.0)

    if denoise:
        smooth = cv2.bilateralFilter((lifted * 255.0).astype(np.uint8), 5, 28, 28).astype(np.float32) / 255.0
        lifted = lifted * 0.82 + smooth * 0.18

    # Restore a touch of crispness after lifting shadows.
    blurred = cv2.GaussianBlur((lifted * 255.0).astype(np.uint8), (0, 0), 1.1).astype(np.float32) / 255.0
    lifted = lifted * 1.05 - blurred * 0.05
    return np.clip(lifted * 255.0, 0, 255).astype(np.uint8)


def process_video(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid input video metadata: {input_path}")
    if args.max_frames > 0:
        total = min(total, args.max_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*args.codec[:4]), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {output_path}")

    try:
        frame_id = 0
        while frame_id < total:
            ok, frame = cap.read()
            if not ok:
                break
            relit = apply_relight(
                frame,
                args.shadow_lift,
                args.midtone_lift,
                args.highlight_protect,
                args.warmth,
                args.saturation,
                args.local_contrast,
                args.denoise,
            )
            writer.write(relit)
            frame_id += 1
            if frame_id == 1 or frame_id % max(1, int(fps) * 5) == 0 or frame_id == total:
                pct = frame_id / max(total, 1) * 100.0
                print(f"[relight] {input_path.name}: {frame_id}/{total} ({pct:5.1f}%)", flush=True)
    finally:
        writer.release()
        cap.release()


def main() -> int:
    args = parse_args()
    apply_preset_defaults(args)
    validate_args(args)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_video_dir = input_dir / args.video_subdir
    output_video_dir = output_dir / args.video_subdir

    videos = list_videos(input_video_dir)
    if args.copy_sidecars:
        copy_sidecars(input_dir, output_dir, args.video_subdir)
    output_video_dir.mkdir(parents=True, exist_ok=True)
    write_profile(output_dir, args, videos)

    for video_path in videos:
        process_video(video_path, output_video_dir / video_path.name, args)

    print(f"Wrote relit videos to {output_video_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
