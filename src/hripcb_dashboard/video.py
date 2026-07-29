"""Frame-by-frame video preprocessing and YOLO inference."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import cv2

from hripcb_preprocessing.candidates import apply_candidate


def _make_browser_compatible(input_path: Path, output_path: Path) -> bool:
    """Transcode OpenCV's temporary MP4 into H.264 when FFmpeg is available."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        input_path.replace(output_path)
        return False

    transcoded_path = output_path.with_name(f"{output_path.stem}.h264.mp4")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(transcoded_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        if transcoded_path.exists():
            transcoded_path.unlink()
        input_path.replace(output_path)
        return False

    transcoded_path.replace(output_path)
    input_path.unlink()
    return True


def process_video(
    input_path: Path,
    output_path: Path,
    model,
    candidate: dict,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device,
    progress_callback=None,
) -> dict:
    """Write an annotated video and return reproducible processing statistics."""

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Video has invalid dimensions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    opencv_output_path = output_path.with_name(f"{output_path.stem}.mp4v.mp4")
    writer = cv2.VideoWriter(
        str(opencv_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError("Could not create MP4 output video")

    frames = 0
    detections = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            processed = apply_candidate(frame, candidate)
            result = model.predict(
                source=processed,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )[0]
            plotted = result.plot()
            writer.write(plotted)
            detections += int(len(result.boxes)) if result.boxes is not None else 0
            frames += 1
            if progress_callback:
                progress_callback(frames, total_frames)
    finally:
        capture.release()
        writer.release()

    browser_compatible = _make_browser_compatible(opencv_output_path, output_path)

    return {
        "frames": frames,
        "fps": float(fps),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "detections": detections,
        "browser_compatible": browser_compatible,
        "output": str(output_path),
    }
