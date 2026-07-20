from pathlib import Path

import yaml


def test_baseline_config_is_shared_and_reproducible():
    config = yaml.safe_load(
        Path("configs/baseline.yaml").read_text(encoding="utf-8")
    )

    assert config["model"] == "yolov8s.pt"
    assert config["imgsz"] == 1024
    assert config["seed"] == 42
    assert config["epochs"] == 100
    assert config["patience"] == 20
    assert config["batch"] == 4
