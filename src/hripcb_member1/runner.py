"""Batch generation and frozen-model evaluation for Member 1."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .evaluation import evaluate_variants
from .filters import apply_bbhe, apply_gaussian_filter
from .metrics import calculate_psnr, calculate_ssim
from .report import build_comparison_grid, write_comparison_html

VARIANTS = ("original", "gaussian", "bbhe", "gaussian_bbhe")


def load_member1_config(path: Path) -> dict:
    """Load and validate the direct-from-original Member 1 configuration."""

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = {
        "split",
        "checkpoint",
        "data_config",
        "imgsz",
        "conf",
        "iou",
        "device",
        "workers",
        "gaussian_kernel_size",
        "gaussian_sigma_x",
        "bbhe_strength",
        "jpeg_quality",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Member 1 config missing keys: {', '.join(missing)}")
    if tuple(config.get("variants", VARIANTS)) != VARIANTS:
        raise ValueError(f"Member 1 variants must be exactly: {', '.join(VARIANTS)}")
    return config


def _source_images(dataset_root: Path, split: str) -> list[Path]:
    image_dir = dataset_root / split / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    return paths


def _write_image(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Could not write image: {path}")


def _save_variant(
    *,
    output_root: Path,
    source: Path,
    variant: str,
    image: np.ndarray,
    clean: np.ndarray,
    quality: int,
    metric_rows: list[dict[str, object]],
    timing_rows: list[dict[str, object]],
    started: float,
) -> None:
    output_path = output_root / "images" / variant / source.name
    if variant == "original":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output_path)
    else:
        _write_image(output_path, image, quality)
    metric_rows.append(
        {
            "source": source.name,
            "variant": variant,
            "psnr": calculate_psnr(clean, image),
            "ssim": calculate_ssim(clean, image),
            "path": str(output_path.relative_to(output_root)),
        }
    )
    timing_rows.append(
        {
            "source": source.name,
            "variant": variant,
            "milliseconds": (time.perf_counter() - started) * 1000.0,
        }
    )


def _copy_label(dataset_root: Path, split: str, source: Path, output_root: Path, variant: str) -> None:
    label = dataset_root / split / "labels" / f"{source.stem}.txt"
    if not label.is_file():
        raise FileNotFoundError(f"Label file not found: {label}")
    destination = output_root / "labels" / variant / label.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label, destination)


def _variant_data_yaml(output_root: Path, variant: str, data_config: Path) -> Path:
    source_config = yaml.safe_load(Path(data_config).read_text(encoding="utf-8")) or {}
    payload = {
        "path": str(output_root.resolve()),
        "train": f"images/{variant}",
        "val": f"images/{variant}",
        "test": f"images/{variant}",
        "nc": source_config.get("nc", 6),
        "names": source_config.get("names", {}),
    }
    path = output_root / "model_eval" / variant / "data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(
    dataset_root: Path,
    output_root: Path,
    sample_name: str | None,
    config: dict,
    *,
    checkpoint: Path | None = None,
    evaluate_model: bool = True,
) -> Path:
    """Generate four direct-from-original variants and optionally evaluate them."""

    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    source_paths = _source_images(dataset_root, config["split"])
    selected_sample = sample_name or source_paths[0].name
    source_names = {path.name for path in source_paths}
    if selected_sample not in source_names:
        raise FileNotFoundError(f"Sample image not found in split: {selected_sample}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison").mkdir(exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    representative_images: dict[str, np.ndarray] = {}
    quality = int(config["jpeg_quality"])
    kernel_size = int(config["gaussian_kernel_size"])
    sigma_x = float(config["gaussian_sigma_x"])
    bbhe_strength = float(config["bbhe_strength"])

    for source in source_paths:
        clean = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if clean is None:
            raise OSError(f"Could not read image: {source}")

        started = time.perf_counter()
        _save_variant(
            output_root=output_root,
            source=source,
            variant="original",
            image=clean,
            clean=clean,
            quality=quality,
            metric_rows=metric_rows,
            timing_rows=timing_rows,
            started=started,
        )
        gaussian_started = time.perf_counter()
        gaussian = apply_gaussian_filter(clean, kernel_size=kernel_size, sigma_x=sigma_x)
        _save_variant(
            output_root=output_root,
            source=source,
            variant="gaussian",
            image=gaussian,
            clean=clean,
            quality=quality,
            metric_rows=metric_rows,
            timing_rows=timing_rows,
            started=gaussian_started,
        )
        bbhe_started = time.perf_counter()
        bbhe = apply_bbhe(clean, strength=bbhe_strength)
        _save_variant(
            output_root=output_root,
            source=source,
            variant="bbhe",
            image=bbhe,
            clean=clean,
            quality=quality,
            metric_rows=metric_rows,
            timing_rows=timing_rows,
            started=bbhe_started,
        )
        combined_started = time.perf_counter()
        combined = apply_bbhe(gaussian, strength=bbhe_strength)
        _save_variant(
            output_root=output_root,
            source=source,
            variant="gaussian_bbhe",
            image=combined,
            clean=clean,
            quality=quality,
            metric_rows=metric_rows,
            timing_rows=timing_rows,
            started=combined_started,
        )

        if source.name == selected_sample:
            representative_images = {
                "Original": clean,
                "Gaussian Filtering": gaussian,
                "BBHE": bbhe,
                "Gaussian + BBHE": combined,
            }

        if evaluate_model:
            for variant in VARIANTS:
                _copy_label(dataset_root, config["split"], source, output_root, variant)

    _write_csv(
        output_root / "image_metrics.csv",
        metric_rows,
        ["source", "variant", "psnr", "ssim", "path"],
    )
    _write_csv(
        output_root / "processing_times.csv",
        timing_rows,
        ["source", "variant", "milliseconds"],
    )

    model_metrics: list[dict[str, object]] = []
    checkpoint_value = checkpoint or config.get("checkpoint")
    checkpoint_path = Path(checkpoint_value).resolve() if checkpoint_value else None
    variant_data: dict[str, Path] = {}
    if evaluate_model:
        data_config = Path(config["data_config"]).resolve()
        if not data_config.is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {data_config}")
        for variant in VARIANTS:
            variant_data[variant] = _variant_data_yaml(output_root, variant, data_config)
        model_metrics = evaluate_variants(
            checkpoint=checkpoint_path,
            variant_data=variant_data,
            output_root=output_root,
            split=config["split"],
            imgsz=int(config["imgsz"]),
            conf=float(config["conf"]),
            iou=float(config["iou"]),
            device=str(config["device"]),
            workers=int(config["workers"]),
        )

    comparison_dir = output_root / "comparison"
    build_comparison_grid(representative_images, comparison_dir / "comparison_grid.jpg")
    panel_names = [
        ("Original", "original", "Original source image; baseline input."),
        ("Gaussian Filtering", "gaussian", "Gaussian Filtering applied directly to the original image."),
        ("BBHE", "bbhe", "Brightness-preserving bi-histogram equalization applied directly to the original image."),
        ("Gaussian + BBHE", "gaussian_bbhe", "Gaussian Filtering followed by BBHE."),
    ]
    panels = [
        {
            "label": label,
            "variant": variant,
            "src": f"../images/{variant}/{selected_sample}",
            "description": description,
        }
        for label, variant, description in panel_names
    ]
    comparison_context = {
        "source": selected_sample,
        "parameters": (
            f"Gaussian kernel={kernel_size}x{kernel_size}; sigmaX={sigma_x}; "
            f"BBHE strength={bbhe_strength}; "
            f"fixed checkpoint={checkpoint_path.name if checkpoint_path else 'not evaluated'}"
        ),
        "panels": panels,
        "model_metrics": model_metrics,
    }
    write_comparison_html(comparison_dir, comparison_context)
    (comparison_dir / "representative_manifest.json").write_text(
        json.dumps(comparison_context, indent=2), encoding="utf-8"
    )
    manifest = {
        "source_count": len(source_paths),
        "split": config["split"],
        "selected_sample": selected_sample,
        "variants": list(VARIANTS),
        "config": config,
        "uses_shared_checkpoint": bool(evaluate_model),
        "checkpoint": str(checkpoint_path) if evaluate_model and checkpoint_path else None,
        "model_metrics": model_metrics,
        "output_root": str(output_root),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_root
