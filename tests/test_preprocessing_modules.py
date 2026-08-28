import cv2
import numpy as np
import pytest

from hripcb_preprocessing.candidates import apply_candidate, build_candidates
from hripcb_preprocessing.filters import apply_homomorphic_filter, apply_wavelet_denoise


def _image():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (36, 34), (180, 120, 60), -1)
    cv2.circle(image, (46, 24), 8, (220, 220, 220), -1)
    return image


MEMBER2_CONFIG = {
    "wavelet_presets": [
        {"id": "w_db1", "wavelet": "db1", "method": "BayesShrink", "mode": "soft"},
        {"id": "w_sym4", "wavelet": "sym4", "method": "BayesShrink", "mode": "soft"},
        {"id": "w_visu", "wavelet": "sym4", "method": "VisuShrink", "mode": "soft", "wavelet_levels": 2},
    ],
    "homomorphic_presets": [
        {"id": "h_c30", "gamma_low": 0.5, "gamma_high": 1.5, "cutoff": 30.0},
        {"id": "h_c50", "gamma_low": 0.7, "gamma_high": 1.3, "cutoff": 50.0},
        {"id": "h_c80", "gamma_low": 0.8, "gamma_high": 1.2, "cutoff": 80.0},
    ],
}


def test_member2_grid_has_original_single_and_combined_candidates():
    candidates = build_candidates("member2", MEMBER2_CONFIG)
    assert len(candidates) == 16
    assert candidates[0]["technique"] == "original"
    assert sum(candidate["technique"] == "wavelet" for candidate in candidates) == 3
    assert sum(candidate["technique"] == "homomorphic" for candidate in candidates) == 3
    assert sum(candidate["technique"] == "wavelet_homomorphic" for candidate in candidates) == 9


def test_member2_candidates_apply_without_shape_changes():
    image = _image()
    for candidate in build_candidates("member2", MEMBER2_CONFIG):
        result = apply_candidate(image, candidate)
        assert result.shape == image.shape
        assert result.dtype == np.uint8


def test_member2_combined_applies_denoise_before_enhancement():
    image = _image()
    combined = next(
        candidate
        for candidate in build_candidates("member2", MEMBER2_CONFIG)
        if candidate["technique"] == "wavelet_homomorphic"
    )
    parameters = combined["parameters"]
    staged = apply_homomorphic_filter(
        apply_wavelet_denoise(
            image,
            wavelet=parameters["wavelet_name"],
            method=parameters["wavelet_method"],
            mode=parameters["wavelet_mode"],
            wavelet_levels=parameters["wavelet_levels"],
        ),
        gamma_low=parameters["homomorphic_gamma_low"],
        gamma_high=parameters["homomorphic_gamma_high"],
        cutoff=parameters["homomorphic_cutoff"],
    )
    assert np.array_equal(apply_candidate(image, combined), staged)


def test_member2_filters_reject_invalid_parameters():
    image = _image()
    with pytest.raises(ValueError):
        apply_wavelet_denoise(image, method="NotAMethod")
    with pytest.raises(ValueError):
        apply_wavelet_denoise(image, wavelet_levels=0)
    with pytest.raises(ValueError):
        apply_homomorphic_filter(image, cutoff=0.0)
    with pytest.raises(ValueError):
        apply_homomorphic_filter(image, gamma_low=1.5, gamma_high=0.5)


@pytest.mark.parametrize("technique", ["median", "clahe", "median_clahe"])
def test_member2_retired_techniques_are_rejected(technique):
    with pytest.raises(ValueError, match=f"Unsupported technique: {technique}"):
        apply_candidate(_image(), {"module": "member2", "technique": technique, "parameters": {}})


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
