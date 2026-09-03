from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from hripcb_preprocessing.candidates import apply_candidate, build_candidates
from hripcb_preprocessing.filters import (
    apply_top_black_hat,
    apply_tv_denoise,
    apply_tv_top_black_hat,
)


def _image():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[..., 1] = np.arange(64, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (36, 34), (180, 120, 60), -1)
    cv2.circle(image, (46, 24), 8, (220, 220, 220), -1)
    return image


MEMBER5_CONFIG = {
    "tv_weights": [0.01, 0.02, 0.05],
    "morphology_kernel_sizes": [5, 9, 15],
    "top_hat_amounts": [0.5, 1.0],
    "black_hat_amounts": [0.5, 1.0],
}


def test_tv_denoise_preserves_shape_and_uint8_output():
    result = apply_tv_denoise(_image(), weight=0.01)

    assert result.shape == _image().shape
    assert result.dtype == np.uint8
    assert np.isfinite(result).all()


@pytest.mark.parametrize("weight", [0, -0.1, float("nan"), float("inf"), True, 1 + 2j])
def test_tv_denoise_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="weight"):
        apply_tv_denoise(_image(), weight=weight)


@pytest.mark.parametrize("shape", [(0, 8, 3), (8, 0, 3)])
def test_new_filters_reject_empty_images(shape):
    image = np.empty(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="non-empty"):
        apply_tv_denoise(image, weight=0.01)
    with pytest.raises(ValueError, match="non-empty"):
        apply_top_black_hat(image, kernel_size=5, top_hat_amount=0.5, black_hat_amount=0.5)
    with pytest.raises(ValueError, match="non-empty"):
        apply_tv_top_black_hat(
            image,
            tv_weight=0.01,
            morphology_kernel_size=5,
            top_hat_amount=0.5,
            black_hat_amount=0.5,
        )


@pytest.mark.parametrize("kernel_size", [0, -1, 4, 5.5, float("nan"), True])
def test_morphology_rejects_invalid_kernel_sizes(kernel_size):
    with pytest.raises(ValueError, match="kernel_size"):
        apply_top_black_hat(
            _image(), kernel_size=kernel_size, top_hat_amount=0.5, black_hat_amount=0.5
        )


@pytest.mark.parametrize("amount_name", ["top_hat_amount", "black_hat_amount"])
@pytest.mark.parametrize("amount", [-0.1, float("nan"), float("inf"), True, 1 + 2j])
def test_morphology_rejects_invalid_amounts(amount_name, amount):
    parameters = {"top_hat_amount": 0.5, "black_hat_amount": 0.5}
    parameters[amount_name] = amount
    with pytest.raises(ValueError, match=amount_name):
        apply_top_black_hat(_image(), kernel_size=5, **parameters)


def test_morphology_primitive_accepts_zero_amounts():
    result = apply_top_black_hat(
        _image(), kernel_size=5, top_hat_amount=0.0, black_hat_amount=0.0
    )

    assert result.shape == _image().shape
    assert result.dtype == np.uint8


def test_morphology_preserves_shape_dtype_and_uses_same_base_for_both_hats(monkeypatch):
    image = _image()
    calls = []
    original = cv2.morphologyEx

    def record(input_image, operation, kernel, *args, **kwargs):
        calls.append((input_image.copy(), operation))
        return original(input_image, operation, kernel, *args, **kwargs)

    monkeypatch.setattr(cv2, "morphologyEx", record)
    result = apply_top_black_hat(
        image, kernel_size=5, top_hat_amount=0.5, black_hat_amount=0.5
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert len(calls) == 2
    assert np.array_equal(calls[0][0], calls[1][0])
    assert {calls[0][1], calls[1][1]} == {cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT}


def test_morphology_combines_signed_contributions_before_clipping(monkeypatch):
    base_luminance = np.array([[250, 5]], dtype=np.uint8)
    ycrcb = np.zeros((1, 2, 3), dtype=np.uint8)
    ycrcb[..., 0] = base_luminance
    ycrcb[..., 1:] = 128
    image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    responses = [
        np.array([[20, 10]], dtype=np.uint8),
        np.array([[10, 20]], dtype=np.uint8),
    ]

    monkeypatch.setattr(cv2, "morphologyEx", lambda *_args, **_kwargs: responses.pop(0))
    result = apply_top_black_hat(image, kernel_size=3, top_hat_amount=1.0, black_hat_amount=1.0)

    result_luminance = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)[..., 0]
    assert np.array_equal(result_luminance, np.array([[255, 0]], dtype=np.uint8))


def test_combined_dispatch_matches_manual_tv_then_morphology_staging():
    image = _image()
    candidate = {
        "module": "member5",
        "technique": "tv_top_black_hat",
        "parameters": {
            "tv_weight": 0.01,
            "morphology_kernel_size": 5,
            "top_hat_amount": 0.5,
            "black_hat_amount": 1.0,
        },
    }
    parameters = candidate["parameters"]
    expected = apply_top_black_hat(
        apply_tv_denoise(image, weight=parameters["tv_weight"]),
        kernel_size=parameters["morphology_kernel_size"],
        top_hat_amount=parameters["top_hat_amount"],
        black_hat_amount=parameters["black_hat_amount"],
    )

    assert np.array_equal(apply_candidate(image, candidate), expected)


def test_member5_grid_has_52_candidates_and_36_combined_candidates():
    candidates = build_candidates("member5", MEMBER5_CONFIG)

    assert len(candidates) == 52
    assert len({candidate["id"] for candidate in candidates}) == 52
    assert candidates[0] == {
        "id": "original",
        "module": "member5",
        "technique": "original",
        "parameters": {},
    }
    assert sum(candidate["technique"] == "tv" for candidate in candidates) == 3
    assert sum(candidate["technique"] == "top_black_hat" for candidate in candidates) == 12
    combined = [candidate for candidate in candidates if candidate["technique"] == "tv_top_black_hat"]
    assert len(combined) == 36
    assert all(
        candidate["parameters"][name] > 0
        for candidate in combined
        for name in ("tv_weight", "top_hat_amount", "black_hat_amount")
    )


def test_member5_config_produces_approved_grid():
    config = yaml.safe_load(Path("configs/member5_full_search.yaml").read_text(encoding="utf-8"))

    candidates = build_candidates("member5", config)
    assert len(candidates) == 52
    assert sum(candidate["technique"] == "tv_top_black_hat" for candidate in candidates) == 36


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("tv_weights", []),
        ("tv_weights", [0.01, 0.01]),
        ("tv_weights", [True]),
        ("tv_weights", [0]),
        ("morphology_kernel_sizes", [4]),
        ("morphology_kernel_sizes", [5, 5.0]),
        ("morphology_kernel_sizes", [5.5]),
        ("top_hat_amounts", [0]),
        ("black_hat_amounts", [0]),
        ("top_hat_amounts", [-0.5]),
        ("black_hat_amounts", [float("nan")]),
    ],
)
def test_member5_grid_rejects_malformed_or_duplicate_values(key, value):
    config = dict(MEMBER5_CONFIG)
    config[key] = value

    with pytest.raises(ValueError, match=key):
        build_candidates("member5", config)


def test_member5_dispatch_rejects_unknown_technique():
    with pytest.raises(ValueError, match="Unsupported technique: unknown"):
        apply_candidate(_image(), {"module": "member5", "technique": "unknown", "parameters": {}})
