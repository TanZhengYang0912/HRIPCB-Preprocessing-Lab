import numpy as np
import pytest

from hripcb_baseline.member3 import (
    add_gaussian_noise,
    apply_member3_pipeline,
    agcwd_luminance,
)


def _sample_rgb() -> np.ndarray:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[..., 0] = np.arange(16, dtype=np.uint8)
    image[..., 1] = 80
    image[..., 2] = 160
    return image


def test_gaussian_noise_is_reproducible_and_clipped():
    image = _sample_rgb()

    first = add_gaussian_noise(image, sigma=25, seed=42)
    second = add_gaussian_noise(image, sigma=25, seed=42)
    different_seed = add_gaussian_noise(image, sigma=25, seed=43)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different_seed)
    assert first.dtype == np.uint8
    assert int(first.min()) >= 0
    assert int(first.max()) <= 255


def test_agcwd_luminance_keeps_constant_image_stable():
    luminance = np.full((8, 8), 127, dtype=np.uint8)

    enhanced = agcwd_luminance(luminance, alpha=0.75)

    assert np.array_equal(enhanced, luminance)


def test_member3_pipeline_preserves_rgb_shape_dtype_and_range():
    image = _sample_rgb()

    processed = apply_member3_pipeline(
        image,
        diameter=5,
        sigma_color=50.0,
        sigma_space=50.0,
        alpha=0.75,
    )

    assert processed.shape == image.shape
    assert processed.dtype == np.uint8
    assert int(processed.min()) >= 0
    assert int(processed.max()) <= 255


def test_member3_rejects_non_rgb_images():
    with pytest.raises(ValueError, match="H x W x 3"):
        apply_member3_pipeline(np.zeros((8, 8), dtype=np.uint8))
