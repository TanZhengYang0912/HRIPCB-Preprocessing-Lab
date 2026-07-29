import cv2
import numpy as np

from hripcb_preprocessing.candidates import apply_candidate, build_candidates


def _image():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (36, 34), (180, 120, 60), -1)
    cv2.circle(image, (46, 24), 8, (220, 220, 220), -1)
    return image


def test_member2_grid_has_original_single_and_combined_candidates():
    config = {
        "median_kernel_sizes": [3, 5, 7],
        "clahe_presets": [
            {"id": "c15", "clip_limit": 1.5, "tile_grid_size": [8, 8]},
            {"id": "c20", "clip_limit": 2.0, "tile_grid_size": [8, 8]},
            {"id": "c30", "clip_limit": 3.0, "tile_grid_size": [8, 8]},
        ],
    }
    candidates = build_candidates("member2", config)
    assert len(candidates) == 16
    assert candidates[0]["technique"] == "original"
    assert sum(candidate["technique"] == "median" for candidate in candidates) == 3
    assert sum(candidate["technique"] == "clahe" for candidate in candidates) == 3
    assert sum(candidate["technique"] == "median_clahe" for candidate in candidates) == 9


def test_member3_and_member4_candidates_apply_without_shape_changes():
    image = _image()
    configs = {
        "member3": {
            "bilateral_presets": [
                {"id": "b05", "diameter": 5, "sigma_color": 25, "sigma_space": 25},
            ],
            "agcwd_gammas": [0.8],
        },
        "member4": {
            "nlm_presets": [
                {"id": "n07", "h": 7, "h_color": 7, "template_window": 7, "search_window": 21},
            ],
            "msr_presets": [
                {"id": "m15250", "sigmas": [15, 80, 250]},
            ],
        },
    }
    for module, config in configs.items():
        candidates = build_candidates(module, config)
        assert len(candidates) == 4
        for candidate in candidates:
            result = apply_candidate(image, candidate)
            assert result.shape == image.shape
            assert result.dtype == np.uint8

