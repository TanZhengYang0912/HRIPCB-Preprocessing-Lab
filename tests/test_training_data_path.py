import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.prepare_final_dataset import prepare
from scripts.train_baseline import resolve_data_path


def test_training_resolves_dataset_yaml_to_a_string_path():
    resolved = resolve_data_path(Path("configs/hripcb_local.yaml"))

    assert isinstance(resolved, str)
    assert Path(resolved).is_file()
    assert Path(resolved).is_absolute()


def test_final_dataset_uses_member2_combined_winner(tmp_path):
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        assert cv2.imwrite(str(image_dir / "sample.jpg"), np.zeros((16, 16, 3), dtype=np.uint8))
        (label_dir / "sample.txt").write_text("", encoding="utf-8")

    selection = {
        "overall_best": {"module": "member1", "technique": "original"},
        "best_by_module": {
            "member2": {"module": "member2", "technique": "original"},
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    data_config = tmp_path / "data.yaml"
    data_config.write_text(yaml.safe_dump({"nc": 1, "names": {0: "defect"}}), encoding="utf-8")

    prepare(dataset, tmp_path / "prepared", selection_path, data_config)

    manifest = json.loads((tmp_path / "prepared" / "preprocessing.json").read_text(encoding="utf-8"))
    assert manifest["candidate"]["module"] == "member2"
