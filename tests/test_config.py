import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_baseline.config import load_config


def test_local_dataset_config_has_six_expected_classes():
    config = load_config(Path("configs/hripcb_local.yaml"))

    assert config["nc"] == 6
    assert config["names"] == [
        "Missing_hole",
        "Mouse_bite",
        "Open_circuit",
        "Short",
        "Spurious_copper",
        "Spur",
    ]


def test_local_dataset_config_points_to_existing_split_directories():
    config = load_config(Path("configs/hripcb_local.yaml"))
    root = Path(config["path"])

    assert (root / config["train"] / "images").is_dir()
    assert (root / config["val"] / "images").is_dir()
    assert (root / config["test"] / "images").is_dir()
