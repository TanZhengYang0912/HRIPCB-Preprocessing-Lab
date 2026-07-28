"""Formal validation-only preprocessing study for Member 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Mapping, Sequence

import cv2
import numpy as np
import yaml

from .member3 import (
    _validate_rgb,
    agcwd_luminance,
    apply_member3_pipeline,
    bilateral_filter_luminance,
)
from .member3_experiment import BilateralConfig


FORMAL_BILATERAL_PRESETS = (
    BilateralConfig(5, 25.0, 25.0),
    BilateralConfig(7, 50.0, 50.0),
    BilateralConfig(9, 75.0, 75.0),
)
FORMAL_GAMMAS = (0.8, 1.0, 1.2)
AGCWD_ALPHA = 0.75
FORMAL_CLASS_NAMES = (
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spurious_copper",
    "Spur",
)
FORMAL_IMAGE_SIZE = 1024
FORMAL_CONFIDENCE = 0.25
FORMAL_IOU = 0.70
FORMAL_WORKERS = 0


@dataclass(frozen=True)
class FormalCondition:
    """One reproducible formal Member 3 preprocessing variant."""

    identifier: str
    technique: str
    bilateral: BilateralConfig | None
    gamma: float

    @property
    def description(self) -> str:
        if self.technique == "original":
            return "Original image"
        if self.technique == "bilateral":
            assert self.bilateral is not None
            return f"Bilateral Filtering ({self.bilateral.tag})"
        if self.technique == "agcwd_gamma":
            return f"AGCWD + gamma={self.gamma:g}"
        assert self.bilateral is not None
        return (
            "Bilateral Filtering + AGCWD "
            f"({self.bilateral.tag}, gamma={self.gamma:g})"
        )


@dataclass(frozen=True)
class ImageQuality:
    """Fidelity metrics between an original and processed RGB image."""

    psnr: float
    ssim: float


@dataclass(frozen=True)
class PreparedCondition:
    """One generated validation dataset and its aggregate image metrics."""

    dataset_root: Path
    data_yaml: Path
    image_count: int
    processing_time_ms: float
    psnr: float
    ssim: float


@dataclass(frozen=True)
class FormalResultRecord:
    """One normalized Member 3 validation result row."""

    condition: FormalCondition
    checkpoint: str
    device: str
    prepared: PreparedCondition
    metrics: Mapping[str, object]

    @classmethod
    def from_metrics(
        cls,
        *,
        condition: FormalCondition,
        checkpoint: str,
        device: str,
        prepared: PreparedCondition,
        metrics: Mapping[str, object],
    ) -> "FormalResultRecord":
        return cls(
            condition=condition,
            checkpoint=checkpoint,
            device=device,
            prepared=prepared,
            metrics=dict(metrics),
        )

    def as_dict(self) -> dict[str, object]:
        bilateral = self.condition.bilateral
        return {
            "model_id": "YOLOv8s baseline",
            "checkpoint": self.checkpoint,
            "member": "Member 3",
            "condition_id": self.condition.identifier,
            "technique": self.condition.technique,
            "training_preprocessing": "None (frozen clean baseline)",
            "evaluation_preprocessing": self.condition.description,
            "dataset_split": "val",
            "validation_images": self.prepared.image_count,
            "imgsz": FORMAL_IMAGE_SIZE,
            "conf": FORMAL_CONFIDENCE,
            "iou": FORMAL_IOU,
            "device": self.device,
            "workers": FORMAL_WORKERS,
            "primary_metric": "mAP50-95",
            "bilateral_diameter": "" if bilateral is None else bilateral.diameter,
            "bilateral_sigma_color": "" if bilateral is None else bilateral.sigma_color,
            "bilateral_sigma_space": "" if bilateral is None else bilateral.sigma_space,
            "agcwd_alpha": "" if self.condition.technique == "original" else AGCWD_ALPHA,
            "gamma": self.condition.gamma,
            "precision": self.metrics.get("precision"),
            "recall": self.metrics.get("recall"),
            "f1": self.metrics.get("f1"),
            "map50": self.metrics.get("map50"),
            "map50_95": self.metrics.get("map50_95"),
            "psnr": self.prepared.psnr,
            "ssim": self.prepared.ssim,
            "processing_time_ms": self.prepared.processing_time_ms,
        }


def build_formal_conditions() -> list[FormalCondition]:
    """Return the agreed original, ablation, and combined 16-condition matrix."""

    conditions = [
        FormalCondition(
            identifier="original",
            technique="original",
            bilateral=None,
            gamma=1.0,
        )
    ]
    conditions.extend(
        FormalCondition(
            identifier=f"bilateral_{preset.tag}",
            technique="bilateral",
            bilateral=preset,
            gamma=1.0,
        )
        for preset in FORMAL_BILATERAL_PRESETS
    )
    conditions.extend(
        FormalCondition(
            identifier=f"agcwd_a{AGCWD_ALPHA:g}_g{gamma:g}",
            technique="agcwd_gamma",
            bilateral=None,
            gamma=gamma,
        )
        for gamma in FORMAL_GAMMAS
    )
    conditions.extend(
        FormalCondition(
            identifier=f"combined_{preset.tag}_a{AGCWD_ALPHA:g}_g{gamma:g}",
            technique="combined",
            bilateral=preset,
            gamma=gamma,
        )
        for preset in FORMAL_BILATERAL_PRESETS
        for gamma in FORMAL_GAMMAS
    )
    return conditions


def _apply_global_gamma(luminance: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    corrected = 255.0 * np.power(luminance.astype(np.float64) / 255.0, gamma)
    return np.rint(corrected).clip(0.0, 255.0).astype(np.uint8)


def apply_formal_condition(
    image: np.ndarray,
    condition: FormalCondition,
) -> np.ndarray:
    """Apply one formal Member 3 condition to an RGB uint8 image."""

    _validate_rgb(image)
    if condition.technique == "original":
        return image.copy()

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    if condition.technique == "bilateral":
        if condition.bilateral is None:
            raise ValueError("bilateral condition requires a bilateral preset")
        ycrcb[..., 0] = bilateral_filter_luminance(
            ycrcb[..., 0],
            diameter=condition.bilateral.diameter,
            sigma_color=condition.bilateral.sigma_color,
            sigma_space=condition.bilateral.sigma_space,
        )
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    if condition.technique == "agcwd_gamma":
        enhanced = agcwd_luminance(ycrcb[..., 0], alpha=AGCWD_ALPHA)
        ycrcb[..., 0] = _apply_global_gamma(enhanced, condition.gamma)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    if condition.technique == "combined":
        if condition.bilateral is None:
            raise ValueError("combined condition requires a bilateral preset")
        enhanced = apply_member3_pipeline(
            image,
            diameter=condition.bilateral.diameter,
            sigma_color=condition.bilateral.sigma_color,
            sigma_space=condition.bilateral.sigma_space,
            alpha=AGCWD_ALPHA,
        )
        enhanced_ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_RGB2YCrCb)
        enhanced_ycrcb[..., 0] = _apply_global_gamma(
            enhanced_ycrcb[..., 0], condition.gamma
        )
        return cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2RGB)

    raise ValueError(f"unknown formal condition technique: {condition.technique}")


def _ssim_channel(first: np.ndarray, second: np.ndarray) -> float:
    first_float = first.astype(np.float64)
    second_float = second.astype(np.float64)
    mean_first = cv2.GaussianBlur(first_float, (11, 11), 1.5)
    mean_second = cv2.GaussianBlur(second_float, (11, 11), 1.5)
    variance_first = cv2.GaussianBlur(first_float * first_float, (11, 11), 1.5)
    variance_first -= mean_first * mean_first
    variance_second = cv2.GaussianBlur(second_float * second_float, (11, 11), 1.5)
    variance_second -= mean_second * mean_second
    covariance = cv2.GaussianBlur(first_float * second_float, (11, 11), 1.5)
    covariance -= mean_first * mean_second
    constant_one = (0.01 * 255.0) ** 2
    constant_two = (0.03 * 255.0) ** 2
    numerator = (2.0 * mean_first * mean_second + constant_one) * (
        2.0 * covariance + constant_two
    )
    denominator = (mean_first * mean_first + mean_second * mean_second + constant_one) * (
        variance_first + variance_second + constant_two
    )
    return float(np.mean(numerator / denominator))


def measure_image_quality(original: np.ndarray, processed: np.ndarray) -> ImageQuality:
    """Return RGB PSNR and mean per-channel SSIM for a processed image."""

    _validate_rgb(original)
    _validate_rgb(processed)
    if original.shape != processed.shape:
        raise ValueError("original and processed images must have the same shape")
    if np.array_equal(original, processed):
        return ImageQuality(psnr=float("inf"), ssim=1.0)
    return ImageQuality(
        psnr=float(cv2.PSNR(original, processed)),
        ssim=float(
            np.mean(
                [
                    _ssim_channel(original[..., channel], processed[..., channel])
                    for channel in range(3)
                ]
            )
        ),
    )


def _validation_image_paths(directory: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _write_validation_yaml(path: Path, dataset_root: Path) -> Path:
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": len(FORMAL_CLASS_NAMES),
        "names": {index: name for index, name in enumerate(FORMAL_CLASS_NAMES)},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def prepare_formal_validation_dataset(
    dataset_root: Path,
    output_root: Path,
    condition: FormalCondition,
) -> PreparedCondition:
    """Generate one formal, processed validation split without changing source data."""

    source_root = Path(dataset_root)
    source_images = source_root / "val" / "images"
    source_labels = source_root / "val" / "labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError("missing validation images or labels directory")
    source_paths = _validation_image_paths(source_images)
    if not source_paths:
        raise ValueError("validation images directory is empty")

    condition_root = Path(output_root) / condition.identifier
    destination_images = condition_root / "val" / "images"
    destination_labels = condition_root / "val" / "labels"
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)

    durations_ms: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for source_path in source_paths:
        label_path = source_labels / f"{source_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing label for {source_path}")
        destination_path = destination_images / source_path.name
        if condition.technique == "original":
            shutil.copy2(source_path, destination_path)
            quality = ImageQuality(psnr=float("inf"), ssim=1.0)
            duration_ms = 0.0
        else:
            image_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"unable to read image: {source_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            processed_rgb = apply_formal_condition(image_rgb, condition)
            duration_ms = (time.perf_counter() - started) * 1000.0
            quality = measure_image_quality(image_rgb, processed_rgb)
            processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(destination_path), processed_bgr):
                raise OSError(f"unable to write image: {destination_path}")
        shutil.copy2(label_path, destination_labels / label_path.name)
        durations_ms.append(duration_ms)
        psnr_values.append(quality.psnr)
        ssim_values.append(quality.ssim)

    data_yaml = _write_validation_yaml(condition_root / "data.yaml", condition_root)
    return PreparedCondition(
        dataset_root=condition_root,
        data_yaml=data_yaml,
        image_count=len(source_paths),
        processing_time_ms=float(np.mean(durations_ms)),
        psnr=float(np.mean(psnr_values)),
        ssim=float(np.mean(ssim_values)),
    )


def rank_formal_results(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return formal rows ordered by the agreed validation primary metric."""

    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (-float(row["map50_95"]), str(row["condition_id"])),
    )
