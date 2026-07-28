#!/usr/bin/env python3
"""Run the formal, validation-only Member 3 preprocessing study."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_baseline.member3_formal import (  # noqa: E402
    FORMAL_CONFIDENCE,
    FORMAL_IMAGE_SIZE,
    FORMAL_IOU,
    FORMAL_WORKERS,
    FormalResultRecord,
    build_formal_conditions,
    prepare_formal_validation_dataset,
    rank_formal_results,
)
from hripcb_baseline.evaluation import select_device, serialise_metrics  # noqa: E402


EXPECTED_VALIDATION_IMAGES = 138
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FormalOutputPaths:
    comparison_csv: Path
    metrics_json: Path
    summary_json: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal Member 3 preprocessing tuning on validation only."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, default=Path("runs/baseline/weights/best.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/member3_formal"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def _validation_image_count(images_directory: Path) -> int:
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in images_directory.iterdir()
    )


def validate_inputs(dataset_root: Path, weights: Path) -> None:
    dataset_root = Path(dataset_root)
    images_directory = dataset_root / "val" / "images"
    labels_directory = dataset_root / "val" / "labels"
    if not images_directory.is_dir() or not labels_directory.is_dir():
        raise FileNotFoundError("missing validation images or labels directory")
    validation_images = _validation_image_count(images_directory)
    if validation_images != EXPECTED_VALIDATION_IMAGES:
        raise ValueError(
            f"expected {EXPECTED_VALIDATION_IMAGES} validation images, "
            f"found {validation_images}"
        )
    if not Path(weights).is_file():
        raise FileNotFoundError(f"checkpoint not found: {weights}")


def write_formal_results(
    output_root: Path,
    rows: Sequence[dict[str, object]],
    *,
    device: str = "auto",
) -> FormalOutputPaths:
    if not rows:
        raise ValueError("formal results must contain at least one row")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ranked_rows = rank_formal_results(rows)
    comparison_csv = output_root / "comparison.csv"
    fields = sorted({key for row in ranked_rows for key in row})
    with comparison_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )

    metrics_json = output_root / "metrics.json"
    metrics_json.write_text(json.dumps(ranked_rows, indent=2), encoding="utf-8")
    best = ranked_rows[0]
    summary_json = output_root / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "member": "Member 3",
                "dataset_split": "val",
                "validation_images": EXPECTED_VALIDATION_IMAGES,
                "checkpoint": best.get("checkpoint"),
                "device": device,
                "imgsz": FORMAL_IMAGE_SIZE,
                "conf": FORMAL_CONFIDENCE,
                "iou": FORMAL_IOU,
                "workers": FORMAL_WORKERS,
                "primary_metric": "mAP50-95",
                "best_condition_id": best["condition_id"],
                "best_map50_95": best["map50_95"],
                "comparison_file": str(comparison_csv),
                "metrics_file": str(metrics_json),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return FormalOutputPaths(comparison_csv, metrics_json, summary_json)


def _evaluate_condition(
    model: Any,
    *,
    data_yaml: Path,
    output_root: Path,
    condition_id: str,
    device: str,
) -> dict[str, object]:
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=FORMAL_IMAGE_SIZE,
        conf=FORMAL_CONFIDENCE,
        iou=FORMAL_IOU,
        device=device,
        project=str(output_root / "evaluation"),
        name=condition_id,
        workers=FORMAL_WORKERS,
        plots=True,
        verbose=True,
    )
    return serialise_metrics(metrics)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    weights = args.weights.resolve()
    output_root = args.output.resolve()
    try:
        validate_inputs(dataset_root, weights)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("Ultralytics is required. Install requirements.txt first.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    device = select_device(args.device)
    model = YOLO(str(weights))
    rows: list[dict[str, object]] = []
    for condition in build_formal_conditions():
        prepared = prepare_formal_validation_dataset(
            dataset_root,
            output_root / "processed",
            condition,
        )
        metrics = _evaluate_condition(
            model,
            data_yaml=prepared.data_yaml,
            output_root=output_root,
            condition_id=condition.identifier,
            device=device,
        )
        rows.append(
            FormalResultRecord.from_metrics(
                condition=condition,
                checkpoint=str(weights),
                device=device,
                prepared=prepared,
                metrics=metrics,
            ).as_dict()
        )

    paths = write_formal_results(output_root, rows, device=device)
    print(f"formal_results: {paths.comparison_csv}")
    print(f"formal_summary: {paths.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
