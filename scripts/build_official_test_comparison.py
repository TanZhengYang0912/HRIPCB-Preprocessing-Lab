#!/usr/bin/env python3
"""Create the official Original/Median/Final test comparison records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f1(metrics: dict) -> float:
    precision = float(metrics.get("metrics/precision(B)", metrics.get("precision", 0.0)))
    recall = float(metrics.get("metrics/recall(B)", metrics.get("recall", 0.0)))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def build(output: Path, baseline_original_metrics: Path, baseline_median_results: Path, final_results: Path) -> Path:
    original = json.loads(Path(baseline_original_metrics).read_text(encoding="utf-8"))
    baseline_median = json.loads(Path(baseline_median_results).read_text(encoding="utf-8"))[0]
    final = json.loads(Path(final_results).read_text(encoding="utf-8"))[0]
    original_metrics = {
        "precision": float(original["metrics/precision(B)"]),
        "recall": float(original["metrics/recall(B)"]),
        "map50": float(original["metrics/mAP50(B)"]),
        "map50_95": float(original["metrics/mAP50-95(B)"]),
        "f1": _f1(original),
    }
    original_record = {
        "id": "baseline_original_test",
        "model_id": "baseline",
        "model_label": "Baseline YOLO",
        "module": "baseline",
        "technique": "original",
        "split": "test",
        "evaluation_type": "official_test",
        "training_preprocessing": "original",
        "evaluation_preprocessing": "original",
        "parameters": {},
        "metrics": original_metrics,
        "protocol": {"imgsz": 1024, "conf": 0.25, "iou": 0.70, "workers": 0},
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = [original_record, baseline_median, final]
    (output / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output / "results.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/official_test_comparison"))
    parser.add_argument("--baseline-original", type=Path, default=Path("runs/evaluation/test/metrics.json"))
    parser.add_argument("--baseline-median", type=Path, default=Path("runs/baseline_median_test/results.json"))
    parser.add_argument("--final", type=Path, default=Path("runs/final_model/evaluation/results.json"))
    args = parser.parse_args()
    print(f"official test records: {build(args.output, args.baseline_original, args.baseline_median, args.final)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
