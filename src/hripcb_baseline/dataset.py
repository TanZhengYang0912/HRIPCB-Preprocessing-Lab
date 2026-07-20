"""Validation utilities for the local HRIPCB YOLO dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_IDS = tuple(range(6))


@dataclass
class DatasetReport:
    """Machine-readable result of validating one HRIPCB dataset root."""

    total_images: int = 0
    total_labels: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[int, int] = field(
        default_factory=lambda: {class_id: 0 for class_id in CLASS_IDS}
    )
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _record_label(report: DatasetReport, label_path: Path) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        report.errors.append(f"Unreadable label {label_path}: {exc}")
        return

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            report.errors.append(
                f"{label_path}:{line_number} must contain 5 YOLO values"
            )
            continue

        try:
            class_id = int(fields[0])
        except ValueError:
            report.errors.append(
                f"{label_path}:{line_number} has an invalid class ID"
            )
            continue

        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            report.errors.append(
                f"{label_path}:{line_number} has non-numeric normalized coordinates"
            )
            continue

        if class_id not in CLASS_IDS:
            report.errors.append(
                f"{label_path}:{line_number} has class ID {class_id}; expected 0-5"
            )
        else:
            report.class_counts[class_id] += 1

        if not all(0.0 <= value <= 1.0 for value in coordinates):
            report.errors.append(
                f"{label_path}:{line_number} has coordinates outside normalized [0, 1]"
            )
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            report.errors.append(
                f"{label_path}:{line_number} has non-positive normalized box size"
            )


def validate_dataset(root: Path) -> DatasetReport:
    """Validate image/label pairs and YOLO annotations under ``root``."""

    root = Path(root)
    report = DatasetReport()

    if not root.is_dir():
        report.errors.append(f"Dataset root does not exist: {root}")
        return report

    for split in SPLITS:
        split_root = root / split
        image_dir = split_root / "images"
        label_dir = split_root / "labels"
        if not image_dir.is_dir():
            report.errors.append(f"Missing image directory: {image_dir}")
        if not label_dir.is_dir():
            report.errors.append(f"Missing label directory: {label_dir}")

        images = _image_files(image_dir)
        labels = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        report.split_counts[split] = len(images)
        report.total_images += len(images)
        report.total_labels += len(labels)

        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        for stem in sorted(image_stems - label_stems):
            report.errors.append(f"Missing label for image: {image_dir / (stem + '.jpg')}")
        for stem in sorted(label_stems - image_stems):
            report.errors.append(f"Label has no matching image: {label_dir / (stem + '.txt')}")

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                report.errors.append(f"Unreadable image {image_path}: {exc}")

        for label_path in labels:
            _record_label(report, label_path)

    return report
