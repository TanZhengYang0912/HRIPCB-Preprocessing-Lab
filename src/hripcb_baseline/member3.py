"""Member 3 preprocessing: bilateral filtering and AGCWD.

The preprocessing operates on the luminance channel only.  This keeps the
colour channels intact while allowing the noise-removal and enhancement
methods to target PCB visibility and contrast.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np


_CHANNEL_COUNT: Final = 3


def _validate_rgb(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != _CHANNEL_COUNT:
        raise ValueError("image must have shape H x W x 3")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")


def _validate_luminance(luminance: np.ndarray) -> None:
    if not isinstance(luminance, np.ndarray):
        raise TypeError("luminance must be a numpy array")
    if luminance.ndim != 2:
        raise ValueError("luminance must have shape H x W")
    if luminance.dtype != np.uint8:
        raise ValueError("luminance must use uint8 pixels")


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add reproducible zero-mean Gaussian noise to the RGB image luminance."""

    _validate_rgb(image)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(sigma), size=ycrcb.shape[:2])
    noisy_y = np.clip(ycrcb[..., 0].astype(np.float32) + noise, 0.0, 255.0)
    output = ycrcb.copy()
    output[..., 0] = np.rint(noisy_y).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB)


def bilateral_filter_luminance(
    luminance: np.ndarray,
    diameter: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    """Apply edge-preserving bilateral filtering to a luminance image."""

    _validate_luminance(luminance)
    if diameter <= 0:
        raise ValueError("diameter must be positive")
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("bilateral sigmas must be positive")
    return cv2.bilateralFilter(
        luminance,
        d=int(diameter),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )


def agcwd_luminance(luminance: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """Enhance luminance with histogram-based adaptive gamma correction.

    The weighted probability distribution is ``p(k) ** alpha``.  Its
    cumulative distribution creates one gamma value per input intensity,
    which is then applied with a lookup table.  A constant image is returned
    unchanged because it contains no contrast information to enhance.
    """

    _validate_luminance(luminance)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in the interval (0, 1]")

    if int(luminance.min()) == int(luminance.max()):
        return luminance.copy()

    histogram = np.bincount(luminance.ravel(), minlength=256).astype(np.float64)
    probability = histogram / float(luminance.size)
    weighted = np.power(probability, float(alpha))
    weighted_sum = float(weighted.sum())
    if weighted_sum == 0.0:
        return luminance.copy()

    weighted_cdf = np.cumsum(weighted) / weighted_sum
    gamma = np.clip(1.0 - weighted_cdf, 0.01, 1.0)
    values = np.arange(256, dtype=np.float64) / 255.0
    lookup = np.rint(255.0 * np.power(values, gamma)).clip(0, 255).astype(np.uint8)
    return lookup[luminance]


def apply_member3_pipeline(
    image: np.ndarray,
    *,
    diameter: int = 5,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
    alpha: float = 0.75,
) -> np.ndarray:
    """Apply Bilateral Filtering followed by AGCWD on the Y channel."""

    _validate_rgb(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    filtered_y = bilateral_filter_luminance(
        ycrcb[..., 0],
        diameter=diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
    )
    ycrcb[..., 0] = agcwd_luminance(filtered_y, alpha=alpha)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
