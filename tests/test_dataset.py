import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_baseline.dataset import validate_dataset


def _make_pair(root: Path, split: str, stem: str, label: str) -> None:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(image_dir / f"{stem}.jpg")
    (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")


def test_validate_dataset_reports_expected_split_counts(tmp_path):
    for split in ("train", "val", "test"):
        _make_pair(tmp_path, split, f"sample_{split}", "0 0.5 0.5 0.25 0.25\n")

    report = validate_dataset(tmp_path)

    assert report.total_images == 3
    assert report.total_labels == 3
    assert report.split_counts == {"train": 1, "val": 1, "test": 1}
    assert report.ok is True


def test_validate_dataset_rejects_invalid_class_and_box(tmp_path):
    _make_pair(tmp_path, "train", "bad", "6 1.2 0.5 0.25 0.25\n")
    for split in ("val", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)

    report = validate_dataset(tmp_path)

    assert report.ok is False
    assert any("class" in error.lower() for error in report.errors)
    assert any("normalized" in error.lower() for error in report.errors)
