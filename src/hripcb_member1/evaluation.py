"""Frozen-checkpoint evaluation for Member 1 preprocessing variants."""

from __future__ import annotations

import csv
import json
import numbers
from pathlib import Path
from typing import Callable


def select_device(requested: str) -> str:
    """Resolve the shared evaluation device without changing model settings."""

    if requested != "auto":
        return requested
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _number(value) -> float | None:
    if isinstance(value, numbers.Number):
        return float(value)
    return None


def _metric_value(metrics, box_key: str, result_keys: tuple[str, ...]) -> float:
    box = getattr(metrics, "box", None)
    if box is not None:
        value = _number(getattr(box, box_key, None))
        if value is not None:
            return value
    results = getattr(metrics, "results_dict", {})
    for key in result_keys:
        value = _number(results.get(key))
        if value is not None:
            return value
    return 0.0


def _metrics_row(variant: str, metrics) -> dict[str, object]:
    precision = _metric_value(metrics, "mp", ("metrics/precision(B)",))
    recall = _metric_value(metrics, "mr", ("metrics/recall(B)",))
    map50 = _metric_value(metrics, "map50", ("metrics/mAP50(B)",))
    map50_95 = _metric_value(metrics, "map", ("metrics/mAP50-95(B)",))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "variant": variant,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "f1": f1,
    }


def evaluate_variants(
    *,
    checkpoint: Path,
    variant_data: dict[str, Path],
    output_root: Path,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    workers: int,
    model_factory: Callable[[str], object] | None = None,
) -> list[dict[str, object]]:
    """Evaluate every variant with one frozen checkpoint and identical settings."""

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if model_factory is None:
        from ultralytics import YOLO

        model_factory = YOLO

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model = model_factory(str(checkpoint))
    resolved_device = select_device(device)
    rows: list[dict[str, object]] = []
    for variant, data_path in variant_data.items():
        metrics = model.val(
            data=str(Path(data_path).resolve()),
            split=split,
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            device=resolved_device,
            project=str(output_root),
            name=variant,
            workers=int(workers),
            plots=False,
            verbose=False,
        )
        rows.append(_metrics_row(variant, metrics))

    fields = ("variant", "precision", "recall", "map50", "map50_95", "f1")
    with (output_root / "model_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "model_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows
