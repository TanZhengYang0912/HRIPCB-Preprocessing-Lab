import cv2
import numpy as np

from hripcb_member1.sweep import build_member1_candidates, apply_candidate


def test_member1_candidate_grid_covers_reasonable_gaussian_bbhe_options():
    config = {
        "gaussian_presets": [
            {"id": "g05_s10", "kernel_size": 5, "sigma_x": 1.0},
            {"id": "g07_s15", "kernel_size": 7, "sigma_x": 1.5},
            {"id": "g09_s20", "kernel_size": 9, "sigma_x": 2.0},
        ],
        "bbhe_strengths": [0.25, 0.5, 0.7, 1.0],
    }

    candidates = build_member1_candidates(config)

    assert len(candidates) == 20
    assert candidates[0]["id"] == "original"
    assert {candidate["technique"] for candidate in candidates} == {
        "original",
        "gaussian",
        "bbhe",
        "gaussian_bbhe",
    }
    combined = [candidate for candidate in candidates if candidate["technique"] == "gaussian_bbhe"]
    assert len(combined) == 12
    assert all("gaussian_kernel_size" in candidate["parameters"] for candidate in combined)
    assert all("bbhe_strength" in candidate["parameters"] for candidate in combined)


def test_apply_candidate_preserves_shape_and_changes_expected_pipeline():
    image = np.full((32, 40, 3), 100, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (30, 24), (220, 220, 220), -1)
    candidate = {
        "technique": "gaussian_bbhe",
        "parameters": {
            "gaussian_kernel_size": 7,
            "gaussian_sigma_x": 1.5,
            "bbhe_strength": 0.5,
        },
    }

    result = apply_candidate(image, candidate)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, image)
