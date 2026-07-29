import csv
import json

import cv2
import numpy as np
import yaml

from hripcb_member1.runner import _variant_data_yaml, run_comparison


def _make_fixture_dataset(root):
    image_dir = root / "test" / "images"
    image_dir.mkdir(parents=True)
    for index in range(2):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[..., 1] = 60 + index * 40
        cv2.rectangle(image, (5, 5), (20, 18), (220, 220, 220), -1)
        path = image_dir / f"sample_{index}.jpg"
        assert cv2.imwrite(str(path), image)
    return image_dir


def test_runner_processes_fixture_images_without_touching_sources(tmp_path):
    image_dir = _make_fixture_dataset(tmp_path / "dataset")
    before = {path: path.read_bytes() for path in image_dir.glob("*.jpg")}
    config = {
        "split": "test",
        "gaussian_kernel_size": 5,
        "gaussian_sigma_x": 1.0,
        "bbhe_strength": 0.5,
        "jpeg_quality": 95,
    }

    output = run_comparison(
        tmp_path / "dataset",
        tmp_path / "output",
        None,
        config,
        evaluate_model=False,
    )

    assert (output / "image_metrics.csv").is_file()
    assert (output / "processing_times.csv").is_file()
    assert (output / "run_manifest.json").is_file()
    with (output / "image_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 4
    assert {row["variant"] for row in rows} == {
        "original",
        "gaussian",
        "bbhe",
        "gaussian_bbhe",
    }
    assert not any("noisy" in row["variant"] for row in rows)
    assert not any("low_contrast" in row["variant"] for row in rows)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["source_count"] == 2
    assert manifest["variants"] == ["original", "gaussian", "bbhe", "gaussian_bbhe"]
    assert manifest["uses_shared_checkpoint"] is False
    assert {path: path.read_bytes() for path in image_dir.glob("*.jpg")} == before


def test_runner_applies_combined_pipeline_after_gaussian(tmp_path):
    _make_fixture_dataset(tmp_path / "dataset")
    config = {
        "split": "test",
        "gaussian_kernel_size": 5,
        "gaussian_sigma_x": 1.0,
        "bbhe_strength": 0.5,
        "jpeg_quality": 95,
    }

    output = run_comparison(
        tmp_path / "dataset",
        tmp_path / "output",
        "sample_0.jpg",
        config,
        evaluate_model=False,
    )

    assert (output / "comparison" / "comparison.html").is_file()
    assert (output / "images" / "gaussian_bbhe" / "sample_0.jpg").is_file()


def test_variant_data_yaml_declares_all_ultralytics_splits(tmp_path):
    source_config = tmp_path / "source.yaml"
    source_config.write_text(
        "nc: 6\nnames: [Missing_hole, Mouse_bite, Open_circuit, Short, Spurious_copper, Spur]\n"
    )

    data_path = _variant_data_yaml(tmp_path / "output", "gaussian", source_config)

    data = yaml.safe_load(data_path.read_text())
    assert data["train"] == "images/gaussian"
    assert data["val"] == "images/gaussian"
    assert data["test"] == "images/gaussian"
