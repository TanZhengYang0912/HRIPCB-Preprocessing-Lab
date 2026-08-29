import cv2
import numpy as np
import pytest

from hripcb_preprocessing.runner import _prepared_image


def test_prepared_image_reuses_existing_file_without_applying_candidate(tmp_path):
    expected = np.full((8, 10, 3), 77, dtype=np.uint8)
    path = tmp_path / "prepared.jpg"
    assert cv2.imwrite(str(path), expected)

    actual = _prepared_image(
        np.zeros_like(expected),
        path,
        {"technique": "unsupported", "parameters": {}},
        reuse=True,
    )

    assert np.array_equal(actual, expected)


def test_prepared_image_reuse_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Prepared image not found"):
        _prepared_image(
            np.zeros((8, 10, 3), dtype=np.uint8),
            tmp_path / "missing.jpg",
            {"technique": "original", "parameters": {}},
            reuse=True,
        )
