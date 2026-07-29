"""OpenCV implementations for Members 2, 3, and 4 preprocessing methods."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have three BGR channels")


def apply_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Remove impulse-like noise while preserving hard PCB edges."""

    _validate_image(image)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return cv2.medianBlur(image, int(kernel_size))


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local luminance contrast without changing chroma."""

    _validate_image(image)
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive")
    if len(tile_grid_size) != 2 or min(tile_grid_size) <= 0:
        raise ValueError("tile_grid_size must contain two positive integers")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_bilateral_filter(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """Smooth within regions while preserving PCB component boundaries."""

    _validate_image(image)
    if diameter <= 0:
        raise ValueError("diameter must be positive")
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("bilateral sigmas must be positive")
    return cv2.bilateralFilter(
        image,
        int(diameter),
        float(sigma_color),
        float(sigma_space),
    )


def apply_agcwd(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply a compact luminance-preserving AGCWD-style enhancement.

    The weighted cumulative distribution produces a pixel-adaptive gamma map;
    ``gamma`` is an explicit global multiplier so the sweep can compare mild,
    standard, and stronger enhancement without changing chroma channels.
    """

    _validate_image(image)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[..., 0].astype(np.float32) / 255.0
    histogram, _ = np.histogram(luminance, bins=256, range=(0.0, 1.0))
    pdf = histogram.astype(np.float32)
    pdf /= max(float(pdf.sum()), 1.0)
    weighted_pdf = pdf / max(float(pdf.max()), 1e-6)
    cdf = np.cumsum(weighted_pdf)
    cdf /= max(float(cdf[-1]), 1e-6)
    indices = np.clip(np.rint(luminance * 255.0), 0, 255).astype(np.int32)
    adaptive_gamma = np.clip(1.0 - cdf[indices], 0.15, 1.35)
    enhanced = np.power(np.clip(luminance, 0.0, 1.0), adaptive_gamma * float(gamma))
    ycrcb[..., 0] = np.clip(np.rint(enhanced * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_non_local_means(
    image: np.ndarray,
    h: float = 7.0,
    h_color: float = 7.0,
    template_window: int = 7,
    search_window: int = 21,
    processing_max_side: int = 768,
) -> np.ndarray:
    """Denoise repeated PCB textures with OpenCV's colored NLM filter."""

    _validate_image(image)
    if h <= 0 or h_color <= 0:
        raise ValueError("NLM h values must be positive")
    if template_window <= 0 or template_window % 2 == 0:
        raise ValueError("template_window must be a positive odd integer")
    if search_window <= 0 or search_window % 2 == 0:
        raise ValueError("search_window must be a positive odd integer")
    if processing_max_side <= 0:
        raise ValueError("processing_max_side must be positive")
    working = image
    scale = 1.0
    if max(image.shape[:2]) > processing_max_side:
        scale = processing_max_side / max(image.shape[:2])
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        working = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    denoised = cv2.fastNlMeansDenoisingColored(
        working,
        None,
        float(h),
        float(h_color),
        int(template_window),
        int(search_window),
    )
    if scale != 1.0:
        return cv2.resize(denoised, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return denoised


def apply_multi_scale_retinex(
    image: np.ndarray,
    sigmas: Sequence[float] = (15.0, 80.0, 250.0),
    processing_max_side: int = 768,
) -> np.ndarray:
    """Apply multi-scale Retinex with percentile normalization per channel."""

    _validate_image(image)
    sigma_values = tuple(float(value) for value in sigmas)
    if not sigma_values or any(value <= 0 for value in sigma_values):
        raise ValueError("sigmas must contain positive values")
    if processing_max_side <= 0:
        raise ValueError("processing_max_side must be positive")
    working = image
    scale = 1.0
    if max(image.shape[:2]) > processing_max_side:
        scale = processing_max_side / max(image.shape[:2])
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        working = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    result = np.empty_like(working)
    for channel_index in range(3):
        channel = working[..., channel_index].astype(np.float32) + 1.0
        retinex = np.zeros_like(channel)
        for sigma in sigma_values:
            blurred = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
            retinex += np.log(channel) - np.log(blurred + 1.0)
        retinex /= len(sigma_values)
        low, high = np.percentile(retinex, (1.0, 99.0))
        if high <= low:
            scaled = np.zeros_like(retinex)
        else:
            scaled = (retinex - low) * 255.0 / (high - low)
        result[..., channel_index] = np.clip(scaled, 0, 255).astype(np.uint8)
    if scale != 1.0:
        return cv2.resize(result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return result
