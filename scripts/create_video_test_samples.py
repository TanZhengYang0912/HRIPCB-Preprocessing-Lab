#!/usr/bin/env python3
"""Create reproducible MP4 test videos from HRIPCB test images.

The videos intentionally do not draw class names on frames. They are for
qualitative video-inference testing, while the source-image manifest preserves
the ground-truth class for later manual comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


CLASSES = (
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spurious_copper",
    "spur",
)


def class_for_image(path: Path) -> str:
    for defect_class in CLASSES:
        if f"_{defect_class}_" in path.name:
            return defect_class
    raise ValueError(f"Could not infer defect class from filename: {path.name}")


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image without distortion and pad it to a fixed video frame."""

    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    resized = cv2.resize(
        image,
        (max(1, round(image_width * scale)), max(1, round(image_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def transcode_to_h264(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to create browser-compatible MP4 videos.")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def make_video(
    name: str,
    images: list[Path],
    output_dir: Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float = 6.0,
    frames_per_image: int = 6,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.mp4"
    manifest_images = []

    with tempfile.TemporaryDirectory(prefix="hripcb_video_sample_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        intermediate_path = temp_dir_path / f"{name}.mp4"
        writer = cv2.VideoWriter(
            str(intermediate_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create intermediate video: {intermediate_path}")

        try:
            for image_path in images:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not read image: {image_path}")
                frame = letterbox(image, width, height)
                for _ in range(frames_per_image):
                    writer.write(frame)
                manifest_images.append({
                    "file": image_path.name,
                    "class": class_for_image(image_path),
                })
        finally:
            writer.release()

        transcode_to_h264(intermediate_path, output_path)

    return {
        "file": output_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": len(images) * frames_per_image,
        "source_images": manifest_images,
    }


def choose_images(image_dir: Path) -> dict[str, list[Path]]:
    by_class = {defect_class: [] for defect_class in CLASSES}
    for path in sorted(image_dir.glob("*.jpg")):
        by_class[class_for_image(path)].append(path)

    missing = [defect_class for defect_class, paths in by_class.items() if not paths]
    if missing:
        raise ValueError(f"Missing test images for classes: {', '.join(missing)}")

    return {
        "mouse_bite_only_test": by_class["mouse_bite"][:6],
        "six_defects_one_each_test": [by_class[defect_class][0] for defect_class in CLASSES],
        "six_defects_mixed_test": [path for defect_class in CLASSES for path in by_class[defect_class][:2]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("HRIPCB_UPDATE/test/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/video_test_samples"))
    args = parser.parse_args()

    selections = choose_images(args.image_dir)
    videos = [make_video(name, images, args.output_dir) for name, images in selections.items()]
    manifest = {
        "dataset_split": "HRIPCB_UPDATE/test/images",
        "purpose": "Qualitative video inference testing; no labels are drawn on frames.",
        "videos": videos,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for video in videos:
        print(f"{video['file']}: {video['frames']} frames, {len(video['source_images'])} source images")
    print(f"manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
