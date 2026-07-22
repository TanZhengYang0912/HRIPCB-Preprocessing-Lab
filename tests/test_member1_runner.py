import csv
import json

import cv2
import numpy as np

from hripcb_member1.runner import run_comparison


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
        "seed": 42,
        "noise_sigmas": [15, 30, 50],
        "contrast_alphas": [0.75, 0.5, 0.25],
        "visual_noise_sigma": 30,
        "visual_contrast_alpha": 0.5,
        "gaussian_kernel_size": 5,
        "gaussian_sigma_x": 1.0,
        "jpeg_quality": 95,
    }

    output = run_comparison(tmp_path / "dataset", tmp_path / "output", None, config)

    assert (output / "image_metrics.csv").is_file()
    assert (output / "processing_times.csv").is_file()
    assert (output / "run_manifest.json").is_file()
    with (output / "image_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 15
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["source_count"] == 2
    assert manifest["uses_shared_checkpoint"] is False
    assert {path: path.read_bytes() for path in image_dir.glob("*.jpg")} == before
