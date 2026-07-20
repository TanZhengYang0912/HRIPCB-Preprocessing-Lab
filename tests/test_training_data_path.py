import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.train_baseline import resolve_data_path


def test_training_resolves_dataset_yaml_to_a_string_path():
    resolved = resolve_data_path(Path("configs/hripcb_local.yaml"))

    assert isinstance(resolved, str)
    assert Path(resolved).is_file()
    assert Path(resolved).is_absolute()
