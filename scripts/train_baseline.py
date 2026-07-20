#!/usr/bin/env python3
"""Train the one shared YOLOv8s HRIPCB baseline model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_baseline.config import load_config


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_data_path(path: Path) -> str:
    """Return a concrete YAML path accepted by Ultralytics."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {resolved}")
    return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {args.config}")

    values = {
        key: value
        for key, value in {
            "data": args.data,
            "model": args.model,
            "project": args.project,
            "name": args.name,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "seed": args.seed,
            "workers": args.workers,
            "device": args.device,
            "amp": args.amp,
        }.items()
        if value is not None
    }
    config.update(values)

    data_path = Path(config["data"])
    data_yaml = resolve_data_path(data_path)
    load_config(data_path)
    project_path = Path(config["project"]).resolve()
    device = select_device(str(config.get("device", "auto")))

    from ultralytics import YOLO

    model = YOLO(str(config["model"]))
    model.train(
        data=data_yaml,
        imgsz=int(config["imgsz"]),
        epochs=int(config["epochs"]),
        patience=int(config["patience"]),
        batch=int(config["batch"]),
        seed=int(config["seed"]),
        workers=int(config["workers"]),
        project=str(project_path),
        name=str(config["name"]),
        device=device,
        amp=bool(config.get("amp", False)),
        deterministic=True,
        cache=False,
        plots=True,
        save=True,
        verbose=True,
    )
    print(f"shared_checkpoint: {project_path / config['name'] / 'weights' / 'best.pt'}")
    print(f"device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
