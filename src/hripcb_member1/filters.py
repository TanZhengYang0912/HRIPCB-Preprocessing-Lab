"""Gaussian Filtering and BBHE implementations for Member 1."""

from __future__ import annotations

import cv2
import numpy as np


def _validate_bgr_uint8(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have three BGR channels")


def _to_ycrcb(image: np.ndarray) -> np.ndarray:
    _validate_bgr_uint8(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


def apply_gaussian_filter(
    image: np.ndarray, kernel_size: int = 5, sigma_x: float = 1.0
) -> np.ndarray:
    """Smooth a BGR image with a fixed-size Gaussian kernel."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma_x <= 0:
        raise ValueError("sigma_x must be positive")
    _validate_bgr_uint8(image)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)


def _equalize_range(values: np.ndarray, low: int, high: int) -> np.ndarray:
    if low > high or values.size == 0 or low == high:
        return values.copy()

    histogram = np.bincount(values, minlength=256)[low : high + 1]
    cumulative = histogram.cumsum()
    nonzero = np.flatnonzero(cumulative)
    if nonzero.size == 0:
        return values.copy()

    cdf_min = int(cumulative[nonzero[0]])
    cdf_max = int(cumulative[-1])
    if cdf_max <= cdf_min:
        return values.copy()

    lut = np.round(
        (cumulative - cdf_min) * (high - low) / (cdf_max - cdf_min) + low
    )
    lut = np.clip(lut, low, high).astype(np.uint8)
    return lut[values - low]


def apply_bbhe(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply weighted brightness-preserving bi-histogram equalization."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    _validate_bgr_uint8(image)
    if strength == 0.0:
        return image.copy()

    ycrcb = _to_ycrcb(image)
    luminance = ycrcb[..., 0]
    mean_value = int(np.mean(luminance))
    output = luminance.copy()

    lower_mask = luminance <= mean_value
    upper_mask = luminance > mean_value
    output[lower_mask] = _equalize_range(
        luminance[lower_mask], 0, mean_value
    )
    output[upper_mask] = _equalize_range(
        luminance[upper_mask], mean_value + 1, 255
    )

    if strength < 1.0:
        blended = (1.0 - strength) * luminance.astype(np.float32) + strength * output
        output = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    ycrcb[..., 0] = output
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
