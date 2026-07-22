"""Gaussian Filtering and BBHE implementations for Member 1."""

from __future__ import annotations

import cv2
import numpy as np

from .degradation import _to_ycrcb, _validate_bgr_uint8


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


def apply_bbhe(image: np.ndarray) -> np.ndarray:
    """Apply brightness-preserving bi-histogram equalization to luminance."""

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

    ycrcb[..., 0] = output
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
