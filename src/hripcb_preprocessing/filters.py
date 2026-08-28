"""OpenCV implementations for Members 2, 3, and 4 preprocessing methods."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from skimage.restoration import denoise_wavelet


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have three BGR channels")


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


WAVELET_METHODS = ("BayesShrink", "VisuShrink")
WAVELET_MODES = ("soft", "hard")


def apply_wavelet_denoise(
    image: np.ndarray,
    wavelet: str = "sym4",
    method: str = "BayesShrink",
    mode: str = "soft",
    wavelet_levels: int | None = None,
) -> np.ndarray:
    """Denoise in the wavelet transform domain by thresholding detail coefficients.

    Unlike the spatial filters used by the other modules, this decomposes the
    image into frequency subbands and shrinks the coefficients that carry noise,
    leaving the coarse structure of the copper tracks untouched.
    """

    _validate_image(image)
    if method not in WAVELET_METHODS:
        raise ValueError(f"method must be one of {WAVELET_METHODS}")
    if mode not in WAVELET_MODES:
        raise ValueError(f"mode must be one of {WAVELET_MODES}")
    if wavelet_levels is not None and int(wavelet_levels) <= 0:
        raise ValueError("wavelet_levels must be a positive integer when provided")
    if not str(wavelet):
        raise ValueError("wavelet must be a non-empty wavelet name")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    denoised = denoise_wavelet(
        rgb,
        channel_axis=-1,
        convert2ycbcr=True,
        method=method,
        mode=mode,
        wavelet=str(wavelet),
        wavelet_levels=None if wavelet_levels is None else int(wavelet_levels),
        rescale_sigma=True,
    )
    denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)


def apply_homomorphic_filter(
    image: np.ndarray,
    gamma_low: float = 0.5,
    gamma_high: float = 1.5,
    cutoff: float = 30.0,
    sharpness: float = 1.0,
) -> np.ndarray:
    """Normalise slowly varying illumination in the log-frequency domain.

    The luminance channel is modelled as illumination multiplied by reflectance.
    Taking the logarithm separates the two into additive low- and high-frequency
    components, so a Gaussian high-pass attenuates the illumination term while
    preserving the reflectance detail that carries the defect signal.
    """

    _validate_image(image)
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if sharpness <= 0:
        raise ValueError("sharpness must be positive")
    if gamma_low < 0:
        raise ValueError("gamma_low must not be negative")
    if gamma_high <= gamma_low:
        raise ValueError("gamma_high must be greater than gamma_low")

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[..., 0].astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(np.log1p(luminance)))

    height, width = luminance.shape
    rows = np.arange(height, dtype=np.float32).reshape(-1, 1) - height / 2.0
    columns = np.arange(width, dtype=np.float32).reshape(1, -1) - width / 2.0
    distance = rows**2 + columns**2
    transfer = (float(gamma_high) - float(gamma_low)) * (
        1.0 - np.exp(-float(sharpness) * distance / (float(cutoff) ** 2))
    ) + float(gamma_low)

    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * transfer)))
    restored = np.expm1(filtered)

    # Rescale by matching the original luminance moments rather than stretching to
    # the full 0-255 range. A percentile stretch dominates the output and destroys
    # far more structure than the frequency filtering itself, which defeats the
    # purpose of illumination normalisation.
    restored_std = float(restored.std())
    if restored_std <= 1e-6:
        scaled = np.full_like(restored, float(luminance.mean()))
    else:
        gain = float(luminance.std()) / restored_std
        scaled = (restored - float(restored.mean())) * gain + float(luminance.mean())
    ycrcb[..., 0] = np.clip(scaled, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
