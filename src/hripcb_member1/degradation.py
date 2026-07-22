"""Deterministic luminance degradations for Member 1 experiments."""

from __future__ import annotations

import cv2
import numpy as np


def _validate_bgr_uint8(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a numpy uint8 array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with three channels")


def _to_ycrcb(image: np.ndarray) -> np.ndarray:
    _validate_bgr_uint8(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


def add_luminance_gaussian_noise(
    image: np.ndarray, sigma: float, seed: int
) -> np.ndarray:
    """Add seeded Gaussian noise to the luminance channel of a BGR image."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    ycrcb = _to_ycrcb(image)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=ycrcb[..., 0].shape)
    luminance = ycrcb[..., 0].astype(np.float32) + noise
    ycrcb[..., 0] = np.clip(luminance, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def reduce_luminance_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """Compress luminance around 128 while preserving chroma."""

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    ycrcb = _to_ycrcb(image)
    luminance = ycrcb[..., 0].astype(np.float32)
    reduced = 128.0 + alpha * (luminance - 128.0)
    ycrcb[..., 0] = np.clip(reduced, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
