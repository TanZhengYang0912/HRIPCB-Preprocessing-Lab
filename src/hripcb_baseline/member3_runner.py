"""Dataset preparation helpers for the Member 3 experiments."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml

from .member3 import (
    add_gaussian_noise,
    agcwd_luminance,
    apply_member3_pipeline,
    bilateral_filter_luminance,
)
from .member3_experiment import BilateralConfig


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONDITIONS = {"clean", "noisy", "bilateral", "agcwd", "member3"}


def _image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _process_condition(
    image: np.ndarray,
    condition: str,
    *,
    sigma: float | None,
    seed: int | None,
    bilateral: BilateralConfig,
    alpha: float,
) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if condition == "clean":
        return image.copy()
    if sigma is None or seed is None:
        raise ValueError(f"{condition} requires sigma and seed")

    noisy = add_gaussian_noise(image, sigma=sigma, seed=seed)
    if condition == "noisy":
        return noisy
    if condition == "member3":
        return apply_member3_pipeline(
            noisy,
            diameter=bilateral.diameter,
            sigma_color=bilateral.sigma_color,
            sigma_space=bilateral.sigma_space,
            alpha=alpha,
        )

    ycrcb = cv2.cvtColor(noisy, cv2.COLOR_RGB2YCrCb)
    if condition == "bilateral":
        ycrcb[..., 0] = bilateral_filter_luminance(
            ycrcb[..., 0],
            diameter=bilateral.diameter,
            sigma_color=bilateral.sigma_color,
            sigma_space=bilateral.sigma_space,
        )
    else:
        ycrcb[..., 0] = agcwd_luminance(ycrcb[..., 0], alpha=alpha)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def prepare_condition_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    split: str,
    condition: str,
    storage_name: str | None = None,
    sigma: float | None = None,
    seed: int | None = None,
    bilateral: BilateralConfig = BilateralConfig(5, 50.0, 50.0),
    alpha: float = 0.75,
) -> Path:
    """Create one condition's YOLO-compatible split without changing source data."""

    source_images = Path(dataset_root) / split / "images"
    source_labels = Path(dataset_root) / split / "labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(f"missing images/labels for split {split}")

    if storage_name is None:
        storage_name = (
            f"{condition}_sigma{sigma:g}"
            if condition != "clean" and sigma is not None
            else condition
        )
    condition_root = Path(output_root) / storage_name / split
    output_images = condition_root / "images"
    output_labels = condition_root / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    for source_image in _image_paths(source_images):
        label_path = source_labels / f"{source_image.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing label for {source_image}")

        destination_image = output_images / source_image.name
        if condition == "clean":
            shutil.copy2(source_image, destination_image)
        else:
            image_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"unable to read image: {source_image}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            processed_rgb = _process_condition(
                image_rgb,
                condition,
                sigma=sigma,
                seed=seed,
                bilateral=bilateral,
                alpha=alpha,
            )
            processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(destination_image), processed_bgr):
                raise OSError(f"unable to write image: {destination_image}")

        shutil.copy2(label_path, output_labels / label_path.name)

    return condition_root


def write_dataset_yaml(
    path: Path,
    dataset_root: Path,
    names: Sequence[str],
) -> Path:
    """Write a local Ultralytics data YAML with the shared class order."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(Path(dataset_root).resolve()),
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": len(names),
        "names": {index: name for index, name in enumerate(names)},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
