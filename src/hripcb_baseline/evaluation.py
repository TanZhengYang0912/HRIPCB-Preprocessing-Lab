"""Shared YOLO evaluation helpers for reproducible experiment runners."""

from __future__ import annotations

import numbers
from typing import Any


def select_device(requested: str) -> str:
    """Use MPS when available for ``auto``; otherwise use CPU."""

    if requested != "auto":
        return requested
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "cpu"


def serialise_metrics(metrics: Any) -> dict[str, object]:
    """Convert an Ultralytics validation result into JSON/CSV-safe values."""

    result: dict[str, object] = {}
    for key, value in getattr(metrics, "results_dict", {}).items():
        if isinstance(value, numbers.Number):
            result[str(key)] = float(value)

    box = getattr(metrics, "box", None)
    if box is None:
        return result

    values = {
        "map50_95": getattr(box, "map", None),
        "map50": getattr(box, "map50", None),
        "precision": getattr(box, "mp", None),
        "recall": getattr(box, "mr", None),
    }
    for key, value in values.items():
        if isinstance(value, numbers.Number):
            result[key] = float(value)

    precision = result.get("precision")
    recall = result.get("recall")
    if isinstance(precision, float) and isinstance(recall, float):
        result["f1"] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    ap = getattr(box, "ap", None)
    if ap is not None:
        result["per_class_ap"] = [float(value) for value in ap]
    return result
