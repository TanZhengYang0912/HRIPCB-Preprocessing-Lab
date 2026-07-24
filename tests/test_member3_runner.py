from pathlib import Path

import numpy as np
from PIL import Image

from hripcb_baseline.member3_experiment import BilateralConfig
from hripcb_baseline.member3_runner import (
    prepare_condition_dataset,
    write_dataset_yaml,
)


def _make_source_split(root: Path) -> None:
    image_dir = root / "val" / "images"
    label_dir = root / "val" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    pixels[..., 0] = 40
    pixels[..., 1] = 90
    pixels[..., 2] = 150
    Image.fromarray(pixels, mode="RGB").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
    )


def test_prepare_condition_dataset_preserves_image_label_pair(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_source_split(source)

    condition_dir = prepare_condition_dataset(
        source,
        output,
        split="val",
        condition="member3",
        sigma=10,
        seed=42,
        bilateral=BilateralConfig(5, 50.0, 50.0),
        alpha=0.75,
    )

    assert (condition_dir / "images/sample.jpg").is_file()
    assert (condition_dir / "labels/sample.txt").read_text() == (
        "0 0.5 0.5 0.25 0.25\n"
    )
    with Image.open(condition_dir / "images/sample.jpg") as image:
        assert image.size == (16, 16)


def test_write_dataset_yaml_uses_absolute_root_and_class_order(tmp_path):
    yaml_path = write_dataset_yaml(
        tmp_path / "generated.yaml",
        tmp_path / "dataset",
        ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spurious_copper", "Spur"],
    )

    text = yaml_path.read_text(encoding="utf-8")
    assert f"path: {str((tmp_path / 'dataset').resolve())}" in text
    assert "nc: 6" in text
    assert "0: Missing_hole" in text
    assert "5: Spur" in text


def test_processed_outputs_keep_different_noise_levels_separate(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_source_split(source)

    sigma10 = prepare_condition_dataset(
        source,
        output,
        split="val",
        condition="noisy",
        sigma=10,
        seed=42,
    )
    sigma25 = prepare_condition_dataset(
        source,
        output,
        split="val",
        condition="noisy",
        sigma=25,
        seed=43,
    )

    assert sigma10 != sigma25
