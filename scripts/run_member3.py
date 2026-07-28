#!/usr/bin/env python3
"""Run the Member 3 Bilateral Filtering + AGCWD experiment."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_baseline.member3_experiment import (  # noqa: E402
    AGCWD_ALPHAS,
    BILATERAL_CANDIDATES,
    NOISE_SEEDS,
    NOISE_SIGMAS,
    BilateralConfig,
    choose_best_member3_parameters,
)
from hripcb_baseline.evaluation import select_device, serialise_metrics  # noqa: E402
from hripcb_baseline.member3_runner import (  # noqa: E402
    prepare_condition_dataset,
    write_dataset_yaml,
)


CLASS_NAMES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spurious_copper",
    "Spur",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, default=Path("runs/baseline/weights/best.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/member3"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Use Bilateral d=5/sigma=50 and AGCWD alpha=0.75 without validation tuning.",
    )
    return parser.parse_args()


def _validate_inputs(dataset_root: Path, weights: Path) -> None:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    for split in ("train", "val", "test"):
        for subdirectory in ("images", "labels"):
            path = dataset_root / split / subdirectory
            if not path.is_dir():
                raise FileNotFoundError(f"Missing dataset directory: {path}")
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")


def _evaluate(
    model: Any,
    data_yaml: Path,
    *,
    split: str,
    project: Path,
    name: str,
    args: argparse.Namespace,
    device: str,
) -> dict[str, object]:
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        project=str(project),
        name=name,
        workers=0,
        plots=True,
        verbose=True,
    )
    return serialise_metrics(metrics)


def _evaluate_baseline(
    model: Any,
    data_yaml: Path,
    args: argparse.Namespace,
    device: str,
    output: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("val", "test"):
        metrics = _evaluate(
            model,
            data_yaml,
            split=split,
            project=output / "evaluation",
            name=f"baseline_{split}",
            args=args,
            device=device,
        )
        rows.append(
            {
                "split": split,
                "condition": "clean",
                "sigma": "",
                "bilateral": "",
                "alpha": "",
                **metrics,
            }
        )
    return rows


def _prepare_and_evaluate(
    model: Any,
    dataset_root: Path,
    output: Path,
    *,
    condition: str,
    sigma: int,
    bilateral: BilateralConfig,
    alpha: float,
    split: str,
    args: argparse.Namespace,
    device: str,
    name: str,
) -> dict[str, object]:
    condition_root = prepare_condition_dataset(
        dataset_root,
        output,
        split=split,
        condition=condition,
        storage_name=f"{condition}_sigma{sigma}",
        sigma=sigma,
        seed=NOISE_SEEDS[sigma],
        bilateral=bilateral,
        alpha=alpha,
    ).parent
    data_yaml = write_dataset_yaml(
        condition_root / "data.yaml", condition_root, CLASS_NAMES
    )
    return _evaluate(
        model,
        data_yaml,
        split=split,
        project=output / "evaluation",
        name=name,
        args=args,
        device=device,
    )


def _tune_parameters(
    model: Any,
    dataset_root: Path,
    output: Path,
    args: argparse.Namespace,
    device: str,
) -> tuple[BilateralConfig, float, dict[str, dict[int, float]]]:
    score_table: dict[tuple[BilateralConfig, float], dict[int, float]] = {}
    tuning_root = output / "tuning"
    for bilateral in BILATERAL_CANDIDATES:
        for alpha in AGCWD_ALPHAS:
            scores: dict[int, float] = {}
            tag = f"{bilateral.tag}_a{alpha:g}"
            for sigma in NOISE_SIGMAS:
                metrics = _prepare_and_evaluate(
                    model,
                    dataset_root,
                    tuning_root / tag,
                    condition="member3",
                    sigma=sigma,
                    bilateral=bilateral,
                    alpha=alpha,
                    split="val",
                    args=args,
                    device=device,
                    name=f"tune_{tag}_sigma{sigma}",
                )
                scores[sigma] = float(metrics["map50_95"])
            score_table[(bilateral, alpha)] = scores
            # Tuning inputs are high-resolution images; keep only the final
            # chosen configuration's processed images and evaluation records.
            shutil.rmtree(tuning_root / tag, ignore_errors=True)

    best_bilateral, best_alpha = choose_best_member3_parameters(score_table)
    serialisable_scores = {
        f"{bilateral.tag}_a{alpha:g}": values
        for (bilateral, alpha), values in score_table.items()
    }
    return best_bilateral, best_alpha, serialisable_scores


def _run_final_conditions(
    model: Any,
    dataset_root: Path,
    output: Path,
    *,
    bilateral: BilateralConfig,
    alpha: float,
    args: argparse.Namespace,
    device: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    processed_root = output / "processed"
    for sigma in NOISE_SIGMAS:
        for condition in ("noisy", "bilateral", "agcwd", "member3"):
            for split in ("val", "test"):
                metrics = _prepare_and_evaluate(
                    model,
                    dataset_root,
                    processed_root,
                    condition=condition,
                    sigma=sigma,
                    bilateral=bilateral,
                    alpha=alpha,
                    split=split,
                    args=args,
                    device=device,
                    name=f"{condition}_sigma{sigma}_{split}",
                )
                rows.append(
                    {
                        "split": split,
                        "condition": condition,
                        "sigma": sigma,
                        "bilateral": bilateral.tag,
                        "alpha": alpha,
                        **metrics,
                    }
                )
    return rows


def _write_results(output: Path, rows: list[dict[str, object]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "metrics.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fields = sorted({key for row in rows for key in row})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialised = {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
            writer.writerow(serialised)


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    try:
        _validate_inputs(dataset_root, weights)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "Ultralytics is required. Install requirements.txt before running.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 2

    output.mkdir(parents=True, exist_ok=True)
    raw_yaml = write_dataset_yaml(output / "raw_data.yaml", dataset_root, CLASS_NAMES)
    device = select_device(args.device)
    model = YOLO(str(weights))
    rows = _evaluate_baseline(model, raw_yaml, args, device, output)

    if args.skip_tuning:
        best_bilateral = BILATERAL_CANDIDATES[1]
        best_alpha = 0.75
        tuning_scores: dict[str, dict[int, float]] = {}
    else:
        best_bilateral, best_alpha, tuning_scores = _tune_parameters(
            model, dataset_root, output, args, device
        )

    rows.extend(
        _run_final_conditions(
            model,
            dataset_root,
            output,
            bilateral=best_bilateral,
            alpha=best_alpha,
            args=args,
            device=device,
        )
    )
    _write_results(output, rows)
    summary = {
        "dataset_root": str(dataset_root),
        "weights": str(weights),
        "device": device,
        "noise_sigmas": list(NOISE_SIGMAS),
        "noise_seeds": NOISE_SEEDS,
        "best_bilateral": best_bilateral.__dict__,
        "best_agcwd_alpha": best_alpha,
        "validation_tuning_scores": tuning_scores,
        "metrics_file": str(output / "metrics.json"),
        "comparison_file": str(output / "comparison.csv"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"member3_output: {output}")
    print(f"best_bilateral: {best_bilateral.tag}")
    print(f"best_agcwd_alpha: {best_alpha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
