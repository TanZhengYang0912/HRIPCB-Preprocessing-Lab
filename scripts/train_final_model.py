#!/usr/bin/env python3
"""Train a preprocessing candidate with the exact shared baseline protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_member1.evaluation import select_device


def train(
    data_path: Path,
    output_root: Path,
    config_path: Path,
    initial_weights: Path,
    epochs_override: int | None = None,
) -> Path:
    from ultralytics import YOLO

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    output_root = Path(output_root).resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    # Use the same initial pretrained model as train_baseline.py. The trained
    # baseline best.pt must not be used for this independent final run.
    model = YOLO(str(initial_weights))
    device = select_device(str(config.get("device", "auto")))
    epochs = int(epochs_override if epochs_override is not None else config.get("epochs", 100))
    model.train(
        data=str(Path(data_path).resolve()),
        imgsz=int(config.get("imgsz", 1024)),
        epochs=epochs,
        patience=int(config.get("patience", 20)),
        batch=int(config.get("batch", 4)),
        seed=int(config.get("seed", 42)),
        workers=int(config.get("workers", 0)),
        amp=bool(config.get("amp", False)),
        device=device,
        deterministic=True,
        cache=False,
        plots=True,
        save=True,
        project=str(output_root.parent),
        name=output_root.name,
        exist_ok=True,
        verbose=True,
    )
    best_path = output_root / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Final best.pt not found: {best_path}")
    (output_root / "final_training_manifest.json").write_text(json.dumps({
        "initial_weights": str(Path(initial_weights).resolve()),
        "data": str(Path(data_path).resolve()),
        "config": {**config, "epochs": epochs},
        "device": device,
        "best_weights": str(best_path),
    }, indent=2), encoding="utf-8")
    return best_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("runs/retrained_median_dataset/data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/retrained_median_candidate"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("yolov8s.pt"), help="Same initial model used by the baseline")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override; use config value 100 for the official run")
    args = parser.parse_args()
    print(f"final weights: {train(args.data, args.output, args.config, args.weights, args.epochs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
