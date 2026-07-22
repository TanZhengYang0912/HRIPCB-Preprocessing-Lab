"""Batch orchestration for Member 1 image comparisons."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .degradation import add_luminance_gaussian_noise, reduce_luminance_contrast
from .filters import apply_bbhe, apply_gaussian_filter
from .metrics import calculate_psnr, calculate_ssim, derive_variant_seed, variant_name
from .report import build_comparison_grid, write_comparison_html


def load_member1_config(path: Path) -> dict:
    """Load and validate the Member 1 YAML configuration."""

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {
        "split",
        "seed",
        "noise_sigmas",
        "contrast_alphas",
        "visual_noise_sigma",
        "visual_contrast_alpha",
        "gaussian_kernel_size",
        "gaussian_sigma_x",
        "jpeg_quality",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Member 1 config missing keys: {', '.join(missing)}")
    return config


def _write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ):
        raise OSError(f"Could not write image: {path}")


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


def _save_variant(
    *,
    output_root: Path,
    source: Path,
    variant: str,
    image: np.ndarray,
    clean: np.ndarray,
    quality: int,
    noise_sigma: float | None,
    contrast_alpha: float | None,
    metric_rows: list[dict[str, object]],
    timing_rows: list[dict[str, object]],
    started: float,
) -> None:
    output_path = output_root / "images" / variant / source.name
    _write_jpeg(output_path, image, quality)
    metric_rows.append(
        {
            "source": source.name,
            "variant": variant,
            "noise_sigma": "" if noise_sigma is None else noise_sigma,
            "contrast_alpha": "" if contrast_alpha is None else contrast_alpha,
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
) -> Path:
    """Process every image in the configured split and write comparison assets."""

    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    source_paths = _source_images(dataset_root, config["split"])
    selected_sample = sample_name or source_paths[0].name
    if selected_sample not in {path.name for path in source_paths}:
        raise FileNotFoundError(f"Sample image not found in split: {selected_sample}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison").mkdir(exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    variants: list[str] = []
    quality = int(config["jpeg_quality"])
    kernel_size = int(config["gaussian_kernel_size"])
    sigma_x = float(config["gaussian_sigma_x"])
    global_seed = int(config["seed"])
    noise_sigmas = [float(value) for value in config["noise_sigmas"]]
    contrast_alphas = [float(value) for value in config["contrast_alphas"]]
    representative_images: dict[str, np.ndarray] = {}

    for source in source_paths:
        clean = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if clean is None:
            raise OSError(f"Could not read image: {source}")
        source_key = source.relative_to(dataset_root).as_posix()

        started = time.perf_counter()
        original_variant = "original"
        _save_variant(
            output_root=output_root,
            source=source,
            variant=original_variant,
            image=clean,
            clean=clean,
            quality=quality,
            noise_sigma=None,
            contrast_alpha=None,
            metric_rows=metric_rows,
            timing_rows=timing_rows,
            started=started,
        )
        if source.name == selected_sample:
            representative_images["Original"] = clean
        variants.append(original_variant) if original_variant not in variants else None

        for sigma in noise_sigmas:
            sigma_label = variant_name("sigma", sigma)
            noisy_variant = f"noisy_{sigma_label}"
            noisy_started = time.perf_counter()
            noisy = add_luminance_gaussian_noise(
                clean,
                sigma=sigma,
                seed=derive_variant_seed(global_seed, source_key, noisy_variant),
            )
            _save_variant(
                output_root=output_root,
                source=source,
                variant=noisy_variant,
                image=noisy,
                clean=clean,
                quality=quality,
                noise_sigma=sigma,
                contrast_alpha=None,
                metric_rows=metric_rows,
                timing_rows=timing_rows,
                started=noisy_started,
            )
            if source.name == selected_sample and sigma == float(config["visual_noise_sigma"]):
                representative_images["Noisy"] = noisy
            gaussian_variant = f"gaussian_{sigma_label}"
            gaussian_started = time.perf_counter()
            gaussian = apply_gaussian_filter(noisy, kernel_size=kernel_size, sigma_x=sigma_x)
            _save_variant(
                output_root=output_root,
                source=source,
                variant=gaussian_variant,
                image=gaussian,
                clean=clean,
                quality=quality,
                noise_sigma=sigma,
                contrast_alpha=None,
                metric_rows=metric_rows,
                timing_rows=timing_rows,
                started=gaussian_started,
            )
            if source.name == selected_sample and sigma == float(config["visual_noise_sigma"]):
                representative_images["Gaussian Filtering"] = gaussian
            variants.extend(name for name in (noisy_variant, gaussian_variant) if name not in variants)

        for alpha in contrast_alphas:
            alpha_label = variant_name("alpha", alpha)
            low_variant = f"low_contrast_{alpha_label}"
            low_started = time.perf_counter()
            low_contrast = reduce_luminance_contrast(clean, alpha=alpha)
            _save_variant(
                output_root=output_root,
                source=source,
                variant=low_variant,
                image=low_contrast,
                clean=clean,
                quality=quality,
                noise_sigma=None,
                contrast_alpha=alpha,
                metric_rows=metric_rows,
                timing_rows=timing_rows,
                started=low_started,
            )
            if source.name == selected_sample and alpha == float(config["visual_contrast_alpha"]):
                representative_images["Low Contrast"] = low_contrast
            bbhe_variant = f"bbhe_{alpha_label}"
            bbhe_started = time.perf_counter()
            bbhe = apply_bbhe(low_contrast)
            _save_variant(
                output_root=output_root,
                source=source,
                variant=bbhe_variant,
                image=bbhe,
                clean=clean,
                quality=quality,
                noise_sigma=None,
                contrast_alpha=alpha,
                metric_rows=metric_rows,
                timing_rows=timing_rows,
                started=bbhe_started,
            )
            if source.name == selected_sample and alpha == float(config["visual_contrast_alpha"]):
                representative_images["BBHE"] = bbhe
            variants.extend(name for name in (low_variant, bbhe_variant) if name not in variants)

        combined_noise = float(config["visual_noise_sigma"])
        combined_alpha = float(config["visual_contrast_alpha"])
        combined_label = f"sigma{int(combined_noise)}_{variant_name('alpha', combined_alpha)}"
        combined_noisy_variant = f"combined_noisy_{combined_label}"
        combined_started = time.perf_counter()
        combined_noisy = reduce_luminance_contrast(
            add_luminance_gaussian_noise(
                clean,
                sigma=combined_noise,
                seed=derive_variant_seed(global_seed, source_key, combined_noisy_variant),
            ),
            alpha=combined_alpha,
        )
        _save_variant(
            output_root=output_root,
            source=source,
            variant=combined_noisy_variant,
            image=combined_noisy,
            clean=clean,
            quality=quality,
            noise_sigma=combined_noise,
            contrast_alpha=combined_alpha,
            metric_rows=metric_rows,
            timing_rows=timing_rows,
            started=combined_started,
        )
        if source.name == selected_sample:
            representative_images["Noisy + Low Contrast"] = combined_noisy
        combined_result_variant = f"combined_gaussian_bbhe_{combined_label}"
        combined_result_started = time.perf_counter()
        combined_result = apply_bbhe(
            apply_gaussian_filter(combined_noisy, kernel_size=kernel_size, sigma_x=sigma_x)
        )
        _save_variant(
            output_root=output_root,
            source=source,
            variant=combined_result_variant,
            image=combined_result,
            clean=clean,
            quality=quality,
            noise_sigma=combined_noise,
            contrast_alpha=combined_alpha,
            metric_rows=metric_rows,
            timing_rows=timing_rows,
            started=combined_result_started,
        )
        if source.name == selected_sample:
            representative_images["Gaussian + BBHE"] = combined_result
        variants.extend(
            name
            for name in (combined_noisy_variant, combined_result_variant)
            if name not in variants
        )

    _write_csv(
        output_root / "image_metrics.csv",
        metric_rows,
        ["source", "variant", "noise_sigma", "contrast_alpha", "psnr", "ssim", "path"],
    )
    comparison_dir = output_root / "comparison"
    grid_order = (
        "Original",
        "Noisy",
        "Gaussian Filtering",
        "Low Contrast",
        "BBHE",
        "Gaussian + BBHE",
    )
    grid_images = {name: representative_images[name] for name in grid_order}
    build_comparison_grid(grid_images, comparison_dir / "comparison_grid.jpg")
    panel_names = [
        ("Original", "original", "Clean source image."),
        ("Noisy", f"noisy_sigma{int(config['visual_noise_sigma'])}", "Controlled Gaussian noise input."),
        ("Gaussian Filtering", f"gaussian_sigma{int(config['visual_noise_sigma'])}", "Gaussian blur after noise injection."),
        ("Low Contrast", f"low_contrast_{variant_name('alpha', float(config['visual_contrast_alpha']))}", "Contrast-reduced input."),
        ("BBHE", f"bbhe_{variant_name('alpha', float(config['visual_contrast_alpha']))}", "Brightness-preserving bi-histogram equalization."),
        ("Gaussian + BBHE", f"combined_gaussian_bbhe_sigma{int(config['visual_noise_sigma'])}_{variant_name('alpha', float(config['visual_contrast_alpha']))}", "Denoising followed by contrast enhancement."),
    ]
    panels = [
        {
            "label": label,
            "src": f"../images/{variant}/{selected_sample}",
            "description": description,
        }
        for label, variant, description in panel_names
    ]
    comparison_context = {
        "source": selected_sample,
        "parameters": f"Gaussian noise sigma={int(config['visual_noise_sigma'])}; contrast alpha={float(config['visual_contrast_alpha']):.2f}; Gaussian kernel={kernel_size}x{kernel_size}; sigmaX={sigma_x}",
        "panels": panels,
    }
    write_comparison_html(comparison_dir, comparison_context)
    (comparison_dir / "representative_manifest.json").write_text(
        json.dumps(comparison_context, indent=2), encoding="utf-8"
    )
    _write_csv(
        output_root / "processing_times.csv",
        timing_rows,
        ["source", "variant", "milliseconds"],
    )
    manifest = {
        "source_count": len(source_paths),
        "split": config["split"],
        "selected_sample": selected_sample,
        "variants": variants,
        "config": config,
        "uses_shared_checkpoint": False,
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
