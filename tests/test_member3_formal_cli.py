import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_member3_formal import (
    parse_args,
    validate_inputs,
    write_formal_results,
)


def _make_dataset(root: Path, image_count: int) -> Path:
    image_dir = root / "val" / "images"
    label_dir = root / "val" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(image_count):
        (image_dir / f"board_{index}.jpg").write_bytes(b"image")
        (label_dir / f"board_{index}.txt").write_text("", encoding="utf-8")
    return root


def test_formal_cli_defaults_are_the_shared_contract():
    args = parse_args(["--dataset-root", "HRIPCB_UPDATE"])

    assert args.weights == Path("runs/baseline/weights/best.pt")
    assert args.output == Path("runs/member3_formal")
    assert args.device == "auto"


def test_formal_cli_runs_directly_from_the_repository_root():
    root = Path(__file__).parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/run_member3_formal.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--dataset-root" in result.stdout


def test_validate_inputs_rejects_wrong_validation_image_count(tmp_path):
    dataset_root = _make_dataset(tmp_path / "dataset", image_count=2)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")

    with pytest.raises(ValueError, match="expected 138 validation images"):
        validate_inputs(dataset_root, weights)


def test_write_formal_results_sorts_by_map50_95_and_selects_best(tmp_path):
    paths = write_formal_results(
        tmp_path,
        [
            {"condition_id": "zeta", "map50_95": 0.4},
            {"condition_id": "alpha", "map50_95": 0.6},
        ],
    )

    with paths.comparison_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))

    assert rows[0]["condition_id"] == "alpha"
    assert summary["best_condition_id"] == "alpha"
    assert summary["dataset_split"] == "val"
