#!/usr/bin/env python3
"""Evaluate the frozen shared HRIPCB checkpoint on a clean split."""

from __future__ import annotations

import argparse
import json
import numbers
import sys
from pathlib import Path

import torch


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/hripcb_local.yaml"))
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=Path("runs/evaluation"))
    return parser.parse_args()


def _serializable_metrics(metrics) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in getattr(metrics, "results_dict", {}).items():
        if isinstance(value, numbers.Number):
            result[str(key)] = float(value)
        else:
            result[str(key)] = str(value)

    box = getattr(metrics, "box", None)
    if box is not None:
        for key in ("map", "map50", "mp", "mr"):
            value = getattr(box, key, None)
            if isinstance(value, numbers.Number):
                result[key] = float(value)
        precision = result.get("mp")
        recall = result.get("mr")
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


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        print(f"Checkpoint not found: {weights}", file=sys.stderr)
        return 2

    data_path = args.data.resolve()
    if not data_path.is_file():
        print(f"Dataset YAML not found: {data_path}", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    model = YOLO(str(weights))
    project = args.project.resolve()
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=select_device(args.device),
        project=str(project),
        name=args.split,
        workers=0,
        plots=True,
        verbose=True,
    )

    output_dir = project / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(_serializable_metrics(metrics), indent=2), encoding="utf-8"
    )
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
