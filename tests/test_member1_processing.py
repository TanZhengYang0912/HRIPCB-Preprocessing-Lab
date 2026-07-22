import cv2
import numpy as np
import pytest

from hripcb_member1.degradation import (
    add_luminance_gaussian_noise,
    reduce_luminance_contrast,
)
from hripcb_member1.filters import apply_bbhe, apply_gaussian_filter


@pytest.fixture
def sample_image():
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[..., 0] = np.arange(40, dtype=np.uint8)
    image[..., 1] = 80
    image[..., 2] = 180
    cv2.rectangle(image, (8, 8), (31, 23), (220, 220, 220), -1)
    return image


def test_noise_is_deterministic_and_bounded(sample_image):
    first = add_luminance_gaussian_noise(sample_image, sigma=30, seed=42)
    second = add_luminance_gaussian_noise(sample_image, sigma=30, seed=42)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    assert int(first.min()) >= 0
    assert int(first.max()) <= 255


def test_contrast_reduction_returns_same_shape_and_dtype(sample_image):
    result = reduce_luminance_contrast(sample_image, alpha=0.5)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8


def test_gaussian_filter_returns_same_shape_and_dtype(sample_image):
    result = apply_gaussian_filter(sample_image)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8


def test_bbhe_handles_constant_image_without_invalid_values():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    result = apply_bbhe(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.isfinite(result).all()


def test_processing_rejects_invalid_inputs(sample_image):
    with pytest.raises(ValueError, match="sigma"):
        add_luminance_gaussian_noise(sample_image, sigma=0, seed=42)
    with pytest.raises(ValueError, match="alpha"):
        reduce_luminance_contrast(sample_image, alpha=0)
    with pytest.raises(ValueError, match="odd"):
        apply_gaussian_filter(sample_image, kernel_size=4)
