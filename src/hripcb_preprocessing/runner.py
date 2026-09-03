"""Generic frozen-checkpoint parameter sweep runner for Members 1-5."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from hripcb_dashboard.dashboard import write_dashboard_html
from hripcb_member1.evaluation import evaluate_variants
from hripcb_member1.metrics import calculate_psnr, calculate_ssim
from hripcb_member1.runner import _source_images

from .candidates import apply_candidate, build_candidates


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _write_image(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Could not write image: {path}")


def _prepared_image(
    clean: np.ndarray,
    path: Path,
    candidate: dict,
    reuse: bool,
) -> np.ndarray:
    if not reuse:
        return apply_candidate(clean, candidate)
    prepared = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if prepared is None:
        raise FileNotFoundError(f"Prepared image not found: {path}")
    return prepared


def _write_preview(path: Path, image: np.ndarray, quality: int) -> None:
    if path.is_symlink() or path.resolve() != path:
        raise ValueError(f"Unsafe preview path: {path}; choose a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".preview-", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    try:
        _write_image(Path(temporary), image, quality)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _metric_view(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downsample only image-quality metrics; detector inputs stay full-size."""

    if max_side <= 0 or max(image.shape[:2]) <= max_side:
        return image
    scale = max_side / max(image.shape[:2])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _write_eval_yaml(output_root: Path, candidate_id: str, data_config: Path) -> Path:
    source = yaml.safe_load(Path(data_config).read_text(encoding="utf-8")) or {}
    relative_images = f"variants/{candidate_id}/images"
    payload = {
        "path": str(output_root.resolve()),
        "train": relative_images,
        "val": relative_images,
        "test": relative_images,
        "nc": source.get("nc", 6),
        "names": source.get("names", {}),
    }
    path = output_root / "model_eval" / candidate_id / "data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_flat_csv(path: Path, records: list[dict]) -> None:
    parameter_keys = sorted({key for record in records for key in record.get("parameters", {})})
    metric_keys = sorted({key for record in records for key in record.get("metrics", {})})
    fields = [
        "id", "model_id", "module", "technique", "split",
        *[f"parameter.{key}" for key in parameter_keys],
        *[f"metric.{key}" for key in metric_keys],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "id": record["id"],
                "model_id": record.get("model_id", "baseline"),
                "module": record["module"],
                "technique": record["technique"],
                "split": record.get("split", ""),
            }
            row.update({f"parameter.{key}": value for key, value in record.get("parameters", {}).items()})
            row.update({f"metric.{key}": value for key, value in record.get("metrics", {}).items()})
            writer.writerow(row)


def run_sweep(
    dataset_root: Path, output_root: Path, config: dict,
    *, candidates: list[dict] | None = None,
) -> Path:
    """Generate, evaluate, and publish a generic candidate matrix."""

    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    module = str(config["module"])
    candidates = build_candidates(module, config) if candidates is None else candidates
    ids = [candidate.get("id", "") for candidate in candidates]
    if not ids or any(
        not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in ids
    ) or len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique, non-empty safe path components")
    output_root.mkdir(parents=True, exist_ok=True)
    sources = _source_images(dataset_root, config["split"])
    sample = config.get("sample") or sources[0].name
    if sample not in {source.name for source in sources}:
        raise FileNotFoundError(f"Sample image not found in split: {sample}")

    model_id = str(config.get("model_id", "baseline"))
    base_records: list[dict] = []
    eval_data: dict[str, Path] = {}
    quality = int(config.get("jpeg_quality", 95))
    metrics_max_side = int(config.get("image_metrics_max_side", 256))
    reuse_prepared = bool(config.get("reuse_prepared", False))
    data_config = Path(config["data_config"]).resolve()
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = candidate["id"]
        started_total = time.perf_counter()
        psnr_values: list[float] = []
        ssim_values: list[float] = []
        preview_image = None
        for source in sources:
            clean = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if clean is None:
                raise OSError(f"Could not read image: {source}")
            image_path = output_root / "variants" / candidate_id / "images" / source.name
            processed = _prepared_image(clean, image_path, candidate, reuse_prepared)
            metric_clean = _metric_view(clean, metrics_max_side)
            metric_processed = _metric_view(processed, metrics_max_side)
            psnr_values.append(calculate_psnr(metric_clean, metric_processed))
            ssim_values.append(calculate_ssim(metric_clean, metric_processed))
            if source.name == sample:
                preview_image = processed
            if not reuse_prepared:
                if candidate["technique"] == "original":
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, image_path)
                else:
                    _write_image(image_path, processed, quality)
            source_label = dataset_root / config["split"] / "labels" / f"{source.stem}.txt"
            if not source_label.is_file():
                raise FileNotFoundError(f"Label file not found: {source_label}")
            label_path = output_root / "variants" / candidate_id / "labels" / source_label.name
            if not reuse_prepared:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_label, label_path)
            elif not label_path.is_file():
                raise FileNotFoundError(f"Prepared label not found: {label_path}")
        if preview_image is None:
            raise RuntimeError(f"No preview image generated for {candidate_id}")
        preview_path = output_root / "previews" / f"{candidate_id}.jpg"
        _write_preview(preview_path, preview_image, quality)
        eval_data[candidate_id] = _write_eval_yaml(output_root, candidate_id, data_config)
        base_records.append({
            **candidate,
            "model_id": model_id,
            "model_label": str(config.get("model_label", "Baseline YOLO")),
            "checkpoint": str(Path(config["checkpoint"]).resolve()),
            "training_preprocessing": str(config.get("training_preprocessing", "original")),
            "evaluation_preprocessing": candidate["technique"],
            "evaluation_type": str(config.get("evaluation_type", "ablation")),
            "split": str(config["split"]),
            "preview": str(preview_path.relative_to(output_root)),
            "source_count": len(sources),
            "image_metrics": {
                "mean_psnr": float(np.mean(psnr_values)),
                "mean_ssim": float(np.mean(ssim_values)),
                "milliseconds": (time.perf_counter() - started_total) * 1000.0,
            },
        })
        print(f"[{module} {index}/{len(candidates)}] prepared {candidate_id}", flush=True)

    model_rows = evaluate_variants(
        checkpoint=Path(config["checkpoint"]).resolve(),
        variant_data=eval_data,
        output_root=output_root / "model_eval",
        split=str(config["split"]),
        imgsz=int(config["imgsz"]),
        conf=float(config["conf"]),
        iou=float(config["iou"]),
        device=str(config["device"]),
        workers=int(config["workers"]),
    )
    metric_by_id = {
        row["variant"]: {key: value for key, value in row.items() if key != "variant"}
        for row in model_rows
    }
    records = [
        {
            "id": record["id"],
            "model_id": record["model_id"],
            "model_label": record["model_label"],
            "module": record["module"],
            "technique": record["technique"],
            "checkpoint": record["checkpoint"],
            "training_preprocessing": record["training_preprocessing"],
            "evaluation_preprocessing": record["evaluation_preprocessing"],
            "evaluation_type": record["evaluation_type"],
            "split": record["split"],
            "parameters": record["parameters"],
            "metrics": {**metric_by_id[record["id"]], **record["image_metrics"]},
            "preview": record["preview"],
            "source_count": record["source_count"],
        }
        for record in base_records
    ]
    (output_root / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    _write_flat_csv(output_root / "results.csv", records)
    write_dashboard_html(
        output_root,
        records,
        title=f"{module.title()} / Preprocessing Parameter Sweep",
        primary_metric=str(config.get("primary_metric", "map50_95")),
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps({
            "module": module,
            "model_id": model_id,
            "split": config["split"],
            "source_count": len(sources),
            "candidate_count": len(records),
            "candidates": [record["id"] for record in records],
            "config": config,
            "uses_shared_checkpoint": model_id == "baseline",
        }, indent=2),
        encoding="utf-8",
    )
    return output_root
