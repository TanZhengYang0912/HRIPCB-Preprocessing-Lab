from pathlib import Path
import py_compile

import numpy as np
import pytest

from hripcb_baseline.member3_demo import (
    CONDITION_LABELS,
    CLASS_NAMES,
    draw_detections,
    filter_detections,
    find_matching_label,
    load_ground_truth,
    load_summary_rows,
    predict_image,
    prepare_condition,
    save_demo_artifacts,
)


def _sample_rgb() -> np.ndarray:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[..., 0] = np.arange(16, dtype=np.uint8)
    image[..., 1] = 80
    image[..., 2] = 160
    image[3:9, 4:12] = [220, 40, 90]
    return image


def test_prepare_condition_supports_all_conditions():
    image = _sample_rgb()

    for condition in CONDITION_LABELS:
        output = prepare_condition(image, condition, sigma=10)

        assert output.shape == image.shape
        assert output.dtype == np.uint8
        assert int(output.min()) >= 0
        assert int(output.max()) <= 255


def test_prepare_condition_clean_returns_an_independent_copy():
    image = _sample_rgb()

    output = prepare_condition(image, "Clean", sigma=10)
    output[0, 0] = [255, 255, 255]

    assert np.array_equal(image, _sample_rgb())
    assert not np.array_equal(output, image)


def test_prepare_condition_noise_is_reproducible():
    image = _sample_rgb()

    first = prepare_condition(image, "Noisy", sigma=25)
    second = prepare_condition(image, "Noisy", sigma=25)

    np.testing.assert_array_equal(first, second)


def test_prepare_condition_rejects_unknown_condition_and_sigma():
    image = _sample_rgb()

    with pytest.raises(ValueError, match="unknown condition"):
        prepare_condition(image, "Unknown", sigma=10)
    with pytest.raises(ValueError, match="sigma"):
        prepare_condition(image, "Noisy", sigma=15)


class _FakeBoxes:
    def __init__(self, boxes, confidences, classes):
        self.xyxy = np.asarray(boxes, dtype=np.float32)
        self.conf = np.asarray(confidences, dtype=np.float32)
        self.cls = np.asarray(classes, dtype=np.float32)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, boxes):
        self.result = _FakeResult(
            _FakeBoxes(
                [box[:4] for box in boxes],
                [box[4] for box in boxes],
                [box[5] for box in boxes],
            )
        )
        self.last_kwargs = None

    def predict(self, *, source, **kwargs):
        self.last_kwargs = kwargs
        return [self.result]


def test_predict_image_uses_fixed_inference_size_and_serialises_boxes():
    model = _FakeModel([(12.2, 8.8, 40.9, 31.4, 0.87, 2)])

    records = predict_image(model, _sample_rgb(), conf=0.25, imgsz=1024)

    assert records == [
        {
            "class_id": 2,
            "class_name": "Open_circuit",
            "confidence": 0.87,
            "xyxy": [12, 9, 41, 31],
        }
    ]
    assert model.last_kwargs == {"imgsz": 1024, "conf": 0.25, "verbose": False}


def test_predict_image_returns_empty_list_for_empty_boxes():
    model = _FakeModel([])

    assert predict_image(model, _sample_rgb()) == []


def test_filter_detections_applies_visualisation_threshold_only():
    detections = [
        {"class_name": "Short", "confidence": 0.2, "xyxy": [0, 0, 1, 1]},
        {"class_name": "Spur", "confidence": 0.8, "xyxy": [1, 1, 2, 2]},
    ]

    assert filter_detections(detections, 0.5) == [detections[1]]


def test_draw_detections_preserves_dimensions_and_marks_box():
    image = _sample_rgb()
    records = [
        {
            "class_id": 0,
            "class_name": "Missing_hole",
            "confidence": 0.9,
            "xyxy": [2, 2, 8, 8],
        }
    ]

    annotated = draw_detections(image, records)

    assert annotated.shape == image.shape
    assert annotated.dtype == image.dtype
    assert not np.array_equal(annotated[2, 2], image[2, 2])


def test_load_ground_truth_converts_normalised_yolo_box(tmp_path):
    label_path = tmp_path / "sample.txt"
    label_path.write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")

    records = load_ground_truth(label_path, (50, 100, 3))

    assert records == [
        {
            "class_id": 2,
            "class_name": CLASS_NAMES[2],
            "xyxy": [40, 15, 60, 35],
        }
    ]


def test_find_matching_label_returns_only_existing_dataset_pair(tmp_path):
    image_path = tmp_path / "val" / "images" / "board.jpg"
    label_path = tmp_path / "val" / "labels" / "board.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    assert find_matching_label(tmp_path, "board.jpg") == label_path
    assert find_matching_label(tmp_path, "missing.jpg") is None


def test_save_demo_artifacts_writes_images_and_metadata(tmp_path):
    image = _sample_rgb()
    metadata = {
        "condition": "Bilateral Filtering",
        "sigma": 10,
        "confidence_threshold": 0.25,
        "model_path": "runs/baseline/weights/best.pt",
        "detections": [],
    }

    output_dir = save_demo_artifacts(
        tmp_path,
        original=image,
        processed=image,
        prediction=image,
        metadata=metadata,
        source_name="board.jpg",
    )

    assert output_dir.is_dir()
    assert (output_dir / "original.png").is_file()
    assert (output_dir / "processed.png").is_file()
    assert (output_dir / "prediction.png").is_file()
    saved_metadata = (output_dir / "metadata.json").read_text(encoding="utf-8")
    assert '"source_name": "board.jpg"' in saved_metadata
    assert '"condition": "Bilateral Filtering"' in saved_metadata


def test_load_summary_rows_filters_test_split(tmp_path):
    csv_path = tmp_path / "comparison.csv"
    csv_path.write_text(
        "condition,map50_95,split\n"
        "clean,0.5,val\n"
        "member3,0.4,test\n",
        encoding="utf-8",
    )

    rows = load_summary_rows(csv_path, split="test")

    assert rows == [{"condition": "member3", "map50_95": "0.4", "split": "test"}]


def test_load_summary_rows_returns_empty_for_missing_csv(tmp_path):
    assert load_summary_rows(tmp_path / "missing.csv") == []


def test_demo_script_compiles():
    script_path = Path(__file__).parents[1] / "scripts" / "member3_demo.py"

    py_compile.compile(str(script_path), doraise=True)
