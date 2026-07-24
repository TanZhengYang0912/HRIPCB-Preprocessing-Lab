"""UI-independent helpers for the local Member 3 demo."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from .member3 import (
    add_gaussian_noise,
    agcwd_luminance,
    apply_member3_pipeline,
    bilateral_filter_luminance,
)
from .member3_experiment import BilateralConfig, NOISE_SEEDS, NOISE_SIGMAS


CONDITION_LABELS = (
    "Clean",
    "Noisy",
    "Bilateral Filtering",
    "AGCWD",
    "Bilateral + AGCWD",
)
CLASS_NAMES = (
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spurious_copper",
    "Spur",
)
DEMO_BILATERAL = BilateralConfig(7, 75.0, 75.0)
DEMO_AGCWD_ALPHA = 0.5


def _validate_rgb_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape H x W x 3")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")


def prepare_condition(
    image: np.ndarray,
    condition: str,
    sigma: int = 10,
) -> np.ndarray:
    """Return an RGB uint8 image using the frozen demo parameters."""

    _validate_rgb_image(image)
    if condition not in CONDITION_LABELS:
        raise ValueError(f"unknown condition: {condition}")
    if sigma not in NOISE_SIGMAS:
        raise ValueError(f"sigma must be one of {NOISE_SIGMAS}")
    if condition == "Clean":
        return image.copy()

    noisy = add_gaussian_noise(image, sigma=sigma, seed=NOISE_SEEDS[sigma])
    if condition == "Noisy":
        return noisy
    if condition == "Bilateral + AGCWD":
        return apply_member3_pipeline(
            noisy,
            diameter=DEMO_BILATERAL.diameter,
            sigma_color=DEMO_BILATERAL.sigma_color,
            sigma_space=DEMO_BILATERAL.sigma_space,
            alpha=DEMO_AGCWD_ALPHA,
        )

    ycrcb = cv2.cvtColor(noisy, cv2.COLOR_RGB2YCrCb)
    if condition == "Bilateral Filtering":
        ycrcb[..., 0] = bilateral_filter_luminance(
            ycrcb[..., 0],
            diameter=DEMO_BILATERAL.diameter,
            sigma_color=DEMO_BILATERAL.sigma_color,
            sigma_space=DEMO_BILATERAL.sigma_space,
        )
    else:
        ycrcb[..., 0] = agcwd_luminance(
            ycrcb[..., 0], alpha=DEMO_AGCWD_ALPHA
        )
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"


def filter_detections(
    detections: Sequence[Mapping[str, object]],
    confidence_threshold: float,
) -> list[dict[str, object]]:
    """Filter already-produced detections for visualisation only."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    return [
        dict(detection)
        for detection in detections
        if float(detection.get("confidence", 0.0)) >= confidence_threshold
    ]


def predict_image(
    model: Any,
    image: np.ndarray,
    *,
    conf: float = 0.25,
    imgsz: int = 1024,
) -> list[dict[str, object]]:
    """Run the frozen detector and return JSON-friendly detections."""

    _validate_rgb_image(image)
    results = model.predict(
        source=image,
        imgsz=int(imgsz),
        conf=float(conf),
        verbose=False,
    )
    if not results:
        return []

    boxes = results[0].boxes
    xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
    confidences = _to_numpy(boxes.conf).reshape(-1)
    class_ids = _to_numpy(boxes.cls).reshape(-1)
    records: list[dict[str, object]] = []
    for coordinates, confidence, class_id in zip(
        xyxy, confidences, class_ids
    ):
        integer_class_id = int(class_id)
        records.append(
            {
                "class_id": integer_class_id,
                "class_name": _class_name(integer_class_id),
                "confidence": float(np.round(float(confidence), 6)),
                "xyxy": [int(value) for value in np.rint(coordinates)],
            }
        )
    return records


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Mapping[str, object]],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw prediction boxes and labels on a copy of an RGB image."""

    _validate_rgb_image(image)
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["xyxy"]]
        class_name = str(detection.get("class_name", "defect"))
        confidence = detection.get("confidence")
        label = class_name
        if confidence is not None:
            label = f"{class_name} {float(confidence):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_y = max(14, y1 - 5)
        cv2.putText(
            annotated,
            label,
            (max(0, x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def find_matching_label(dataset_root: Path, source_name: str) -> Path | None:
    """Return a label path only when the image/label pair exists in a split."""

    dataset_root = Path(dataset_root)
    for split in ("train", "val", "test"):
        image_path = dataset_root / split / "images" / source_name
        label_path = dataset_root / split / "labels" / f"{Path(source_name).stem}.txt"
        if image_path.is_file() and label_path.is_file():
            return label_path
    return None


def load_ground_truth(
    label_path: Path,
    image_shape: tuple[int, int, int],
) -> list[dict[str, object]]:
    """Load normalized YOLO labels as pixel-space ground-truth records."""

    label_path = Path(label_path)
    if not label_path.is_file():
        return []
    height, width = image_shape[:2]
    records: list[dict[str, object]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            class_id = int(fields[0])
            center_x, center_y, box_width, box_height = (
                float(value) for value in fields[1:]
            )
        except ValueError:
            continue
        x1 = int(round((center_x - box_width / 2.0) * width))
        y1 = int(round((center_y - box_height / 2.0) * height))
        x2 = int(round((center_x + box_width / 2.0) * width))
        y2 = int(round((center_y + box_height / 2.0) * height))
        records.append(
            {
                "class_id": class_id,
                "class_name": _class_name(class_id),
                "xyxy": [
                    max(0, min(width - 1, x1)),
                    max(0, min(height - 1, y1)),
                    max(0, min(width - 1, x2)),
                    max(0, min(height - 1, y2)),
                ],
            }
        )
    return records


def save_demo_artifacts(
    output_root: Path,
    *,
    original: np.ndarray,
    processed: np.ndarray,
    prediction: np.ndarray,
    metadata: Mapping[str, object],
    source_name: str,
) -> Path:
    """Save the images and JSON metadata for one interactive demo run."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / stem
    suffix = 2
    while output_dir.exists():
        output_dir = output_root / f"{stem}-{suffix}"
        suffix += 1
    output_dir.mkdir()

    for filename, array in (
        ("original.png", original),
        ("processed.png", processed),
        ("prediction.png", prediction),
    ):
        _validate_rgb_image(array)
        Image.fromarray(array).save(output_dir / filename)

    payload = dict(metadata)
    payload["source_name"] = source_name
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return output_dir


def load_summary_rows(
    csv_path: Path,
    *,
    split: str | None = "test",
) -> list[dict[str, str]]:
    """Load the batch summary CSV, optionally filtering by split."""

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if split is None:
        return rows
    return [row for row in rows if row.get("split") == split]
