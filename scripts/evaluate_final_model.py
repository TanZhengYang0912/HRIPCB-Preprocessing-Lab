#!/usr/bin/env python3
"""Evaluate a retrained candidate with the shared project protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_member1.evaluation import select_device


def _value(metrics, attr: str, result_key: str) -> float:
    box = getattr(metrics, "box", None)
    value = getattr(box, attr, None) if box is not None else None
    if isinstance(value, (int, float)):
        return float(value)
    return float(getattr(metrics, "results_dict", {}).get(result_key, 0.0))


def evaluate(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    checkpoint = Path(args.checkpoint).resolve()
    data = Path(args.data).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    metrics = YOLO(str(checkpoint)).val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        workers=args.workers,
        project=str(output),
        name="test_run",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    precision = _value(metrics, "mp", "metrics/precision(B)")
    recall = _value(metrics, "mr", "metrics/recall(B)")
    map50 = _value(metrics, "map50", "metrics/mAP50(B)")
    map50_95 = _value(metrics, "map", "metrics/mAP50-95(B)")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    row = {
        "id": args.record_id or f"{args.model_id}_{args.technique}_{args.split}",
        "model_id": args.model_id,
        "model_label": args.model_label,
        "module": args.module,
        "technique": args.technique,
        "split": args.split,
        "evaluation_type": args.evaluation_type,
        "training_preprocessing": args.training_preprocessing,
        "evaluation_preprocessing": args.evaluation_preprocessing,
        "parameters": {
            "wavelet_name": args.wavelet_name,
            "wavelet_method": args.wavelet_method,
            "wavelet_mode": args.wavelet_mode,
            "wavelet_levels": args.wavelet_levels,
            "homomorphic_gamma_low": args.homomorphic_gamma_low,
            "homomorphic_gamma_high": args.homomorphic_gamma_high,
            "homomorphic_cutoff": args.homomorphic_cutoff,
            "homomorphic_sharpness": args.homomorphic_sharpness,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "f1": f1,
        },
        "protocol": {
            "data": str(data),
            "checkpoint": str(checkpoint),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": device,
            "workers": args.workers,
        },
    }
    (output / "metrics.json").write_text(json.dumps(row["metrics"], indent=2), encoding="utf-8")
    (output / "results.json").write_text(json.dumps([row], indent=2), encoding="utf-8")
    return output / "results.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/retrained_wavelet_homomorphic_candidate/weights/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("runs/retrained_wavelet_homomorphic_dataset/data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/retrained_wavelet_homomorphic_candidate/evaluation"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model-id", default="retrained_wavelet_homomorphic_candidate")
    parser.add_argument("--model-label", default="Retrained Wavelet + Homomorphic Candidate")
    parser.add_argument("--module", default="member2")
    parser.add_argument("--technique", default="wavelet_homomorphic")
    parser.add_argument("--training-preprocessing", default="wavelet_homomorphic")
    parser.add_argument("--evaluation-preprocessing", default="wavelet_homomorphic")
    parser.add_argument("--evaluation-type", default="retrained_candidate")
    parser.add_argument("--wavelet-name", default="coif2")
    parser.add_argument("--wavelet-method", default="VisuShrink")
    parser.add_argument("--wavelet-mode", default="soft")
    parser.add_argument("--wavelet-levels", type=int, default=None)
    parser.add_argument("--homomorphic-gamma-low", type=float, default=0.7)
    parser.add_argument("--homomorphic-gamma-high", type=float, default=1.3)
    parser.add_argument("--homomorphic-cutoff", type=float, default=20.0)
    parser.add_argument("--homomorphic-sharpness", type=float, default=2.0)
    parser.add_argument("--record-id", default="")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    print(f"retrained candidate evaluation: {evaluate(args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
