"""Run Member 1's parameter matrix and publish generic dashboard records."""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from hripcb_dashboard.dashboard import write_dashboard_html

from .evaluation import evaluate_variants
from .metrics import calculate_psnr, calculate_ssim
from .runner import _source_images
from .sweep import apply_candidate, build_member1_candidates


def load_sweep_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _write_image(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
        raise OSError(f"Could not write image: {path}")


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
        "id",
        "module",
        "technique",
        *[f"parameter.{key}" for key in parameter_keys],
        *[f"metric.{key}" for key in metric_keys],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {"id": record["id"], "module": record["module"], "technique": record["technique"]}
            row.update({f"parameter.{key}": value for key, value in record.get("parameters", {}).items()})
            row.update({f"metric.{key}": value for key, value in record.get("metrics", {}).items()})
            writer.writerow(row)


def run_member1_sweep(dataset_root: Path, output_root: Path, config: dict) -> Path:
    """Generate, evaluate, and publish the Member 1 candidate matrix."""

    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = build_member1_candidates(config)
    sources = _source_images(dataset_root, config["split"])
    sample = config.get("sample") or sources[0].name
    if sample not in {source.name for source in sources}:
        raise FileNotFoundError(f"Sample image not found in split: {sample}")

    base_records: list[dict] = []
    eval_data: dict[str, Path] = {}
    quality = int(config["jpeg_quality"])
    data_config = Path(config["data_config"]).resolve()
    for candidate in candidates:
        candidate_id = candidate["id"]
        started_total = time.perf_counter()
        psnr_values: list[float] = []
        ssim_values: list[float] = []
        preview_image = None
        for source in sources:
            clean = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if clean is None:
                raise OSError(f"Could not read image: {source}")
            processed = apply_candidate(clean, candidate)
            psnr_values.append(calculate_psnr(clean, processed))
            ssim_values.append(calculate_ssim(clean, processed))
            if source.name == sample:
                preview_image = processed
            image_path = output_root / "variants" / candidate_id / "images" / source.name
            if candidate["technique"] == "original":
                image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, image_path)
            else:
                _write_image(image_path, processed, quality)
            source_label = dataset_root / config["split"] / "labels" / f"{source.stem}.txt"
            if not source_label.is_file():
                raise FileNotFoundError(f"Label file not found: {source_label}")
            label_path = output_root / "variants" / candidate_id / "labels" / source_label.name
            label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_label, label_path)
        if preview_image is None:
            raise RuntimeError(f"No preview image generated for {candidate_id}")
        preview_path = output_root / "previews" / f"{candidate_id}.jpg"
        _write_image(preview_path, preview_image, quality)
        eval_data[candidate_id] = _write_eval_yaml(output_root, candidate_id, data_config)
        base_records.append(
            {
                **candidate,
                "preview": str(preview_path.relative_to(output_root)),
                "source_count": len(sources),
                "image_metrics": {
                    "mean_psnr": float(np.mean(psnr_values)),
                    "mean_ssim": float(np.mean(ssim_values)),
                    "milliseconds": (time.perf_counter() - started_total) * 1000.0,
                },
            }
        )

    model_rows = evaluate_variants(
        checkpoint=Path(config["checkpoint"]).resolve(),
        variant_data=eval_data,
        output_root=output_root / "model_eval",
        split=config["split"],
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
            "module": record["module"],
            "technique": record["technique"],
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
        title="Preprocessing Parameter Sweep",
        primary_metric=str(config.get("primary_metric", "map50_95")),
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "module": "member1",
                "split": config["split"],
                "candidate_count": len(records),
                "candidates": [record["id"] for record in records],
                "config": config,
                "uses_shared_checkpoint": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_root
