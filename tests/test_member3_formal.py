import numpy as np
from PIL import Image

from hripcb_baseline.member3_formal import (
    FORMAL_GAMMAS,
    FormalResultRecord,
    apply_formal_condition,
    build_formal_conditions,
    measure_image_quality,
    prepare_formal_validation_dataset,
    rank_formal_results,
)


def _sample_rgb() -> np.ndarray:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = np.arange(16, dtype=np.uint8)
    image[..., 1] = 80
    image[..., 2] = 160
    image[4:12, 4:12] = [200, 60, 20]
    return image


def test_build_formal_conditions_contains_the_agreed_sixteen_variants():
    conditions = build_formal_conditions()

    assert len(conditions) == 16
    assert [item.technique for item in conditions].count("original") == 1
    assert [item.technique for item in conditions].count("bilateral") == 3
    assert [item.technique for item in conditions].count("agcwd_gamma") == 3
    assert [item.technique for item in conditions].count("combined") == 9
    assert FORMAL_GAMMAS == (0.8, 1.0, 1.2)


def test_original_condition_returns_an_independent_rgb_copy():
    image = _sample_rgb()

    output = apply_formal_condition(image, build_formal_conditions()[0])
    output[0, 0] = [255, 255, 255]

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert image[0, 0].tolist() != [255, 255, 255]


def test_every_formal_condition_preserves_rgb_shape_and_dtype():
    image = _sample_rgb()

    for condition in build_formal_conditions():
        output = apply_formal_condition(image, condition)

        assert output.shape == image.shape
        assert output.dtype == np.uint8


def test_gamma_one_is_identity_after_agcwd_output():
    image = _sample_rgb()
    gamma_one = next(
        condition
        for condition in build_formal_conditions()
        if condition.technique == "agcwd_gamma" and condition.gamma == 1.0
    )

    output = apply_formal_condition(image, gamma_one)

    assert output.shape == image.shape


def test_measure_image_quality_reports_identity_values_for_identical_images():
    image = _sample_rgb()

    quality = measure_image_quality(image, image.copy())

    assert quality.psnr == float("inf")
    assert quality.ssim == 1.0


def _make_validation_dataset(root, image_count: int = 2):
    image_dir = root / "val" / "images"
    label_dir = root / "val" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(image_count):
        image = _sample_rgb().copy()
        image[..., 0] = np.clip(image[..., 0] + index, 0, 255)
        Image.fromarray(image).save(image_dir / f"board_{index}.png")
        (label_dir / f"board_{index}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )
    return root


def test_prepare_formal_validation_dataset_copies_labels_and_aggregates_metrics(
    tmp_path,
):
    source = _make_validation_dataset(tmp_path / "source")

    prepared = prepare_formal_validation_dataset(
        source,
        tmp_path / "output",
        build_formal_conditions()[1],
    )

    assert (prepared.dataset_root / "val/images/board_0.png").is_file()
    assert (prepared.dataset_root / "val/labels/board_0.txt").read_text() == (
        "0 0.5 0.5 0.25 0.25\n"
    )
    assert prepared.image_count == 2
    assert prepared.processing_time_ms >= 0.0
    assert prepared.psnr > 0.0
    assert 0.0 <= prepared.ssim <= 1.0


def test_formal_result_record_contains_the_team_result_schema(tmp_path):
    source = _make_validation_dataset(tmp_path / "source")
    prepared = prepare_formal_validation_dataset(
        source,
        tmp_path / "output",
        build_formal_conditions()[0],
    )

    record = FormalResultRecord.from_metrics(
        condition=build_formal_conditions()[0],
        checkpoint="runs/baseline/weights/best.pt",
        device="cpu",
        prepared=prepared,
        metrics={
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
            "map50": 0.8,
            "map50_95": 0.5,
        },
    ).as_dict()

    assert record["member"] == "Member 3"
    assert record["dataset_split"] == "val"
    assert record["primary_metric"] == "mAP50-95"
    assert record["gamma"] == 1.0
    assert record["psnr"] == float("inf")


def test_rank_formal_results_uses_map50_95_then_condition_identifier():
    rows = [
        {"condition_id": "zeta", "map50_95": 0.6},
        {"condition_id": "alpha", "map50_95": 0.6},
        {"condition_id": "middle", "map50_95": 0.4},
    ]

    ranked = rank_formal_results(rows)

    assert [row["condition_id"] for row in ranked] == ["alpha", "zeta", "middle"]
