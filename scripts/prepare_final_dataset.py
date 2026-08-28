#!/usr/bin/env python3
"""Materialize all splits with the selected preprocessing for final training."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_member1.runner import _source_images
from hripcb_preprocessing.candidates import apply_candidate


def prepare(dataset_root: Path, output_root: Path, selection_path: Path, data_config: Path) -> Path:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    winner = selection["best_by_module"]["member2"]
    candidate = {
        "module": winner["module"],
        "technique": winner["technique"],
        "parameters": winner.get("parameters", {}),
    }
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_config = yaml.safe_load(Path(data_config).read_text(encoding="utf-8")) or {}
    for split in ("train", "val", "test"):
        for source in _source_images(dataset_root, split):
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Could not read image: {source}")
            processed = apply_candidate(image, candidate)
            image_path = output_root / split / "images" / source.name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if candidate["technique"] == "original":
                shutil.copy2(source, image_path)
            elif not cv2.imwrite(str(image_path), processed, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not write image: {image_path}")
            label_source = dataset_root / split / "labels" / f"{source.stem}.txt"
            label_target = output_root / split / "labels" / label_source.name
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_source, label_target)
    data_payload = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": source_config.get("nc", 6),
        "names": source_config.get("names", {}),
    }
    data_path = output_root / "data.yaml"
    data_path.write_text(yaml.safe_dump(data_payload, sort_keys=False), encoding="utf-8")
    (output_root / "preprocessing.json").write_text(json.dumps({
        "source_dataset": str(dataset_root),
        "candidate": candidate,
        "selection": str(Path(selection_path).resolve()),
        "applied_to": ["train", "val", "test"],
    }, indent=2), encoding="utf-8")
    return data_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("HRIPCB_UPDATE"))
    parser.add_argument("--output", type=Path, default=Path("runs/retrained_wavelet_homomorphic_dataset"))
    parser.add_argument("--selection", type=Path, default=Path("runs/project_validation_comparison/selection.json"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/hripcb_local.yaml"))
    args = parser.parse_args()
    print(f"final data: {prepare(args.dataset, args.output, args.selection, args.data_config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
