"""Image-quality metrics and stable variant identifiers."""

from __future__ import annotations

import hashlib

import numpy as np
from skimage.metrics import structural_similarity


def _validate_pair(reference: np.ndarray, candidate: np.ndarray) -> None:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    if reference.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError("reference and candidate must be uint8 images")


def derive_variant_seed(global_seed: int, relative_name: str, variant: str) -> int:
    """Return a stable 32-bit seed for an image/variant pair."""

    payload = f"{global_seed}|{relative_name}|{variant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def calculate_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Calculate peak signal-to-noise ratio for two uint8 images."""

    _validate_pair(reference, candidate)
    error = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(error * error))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def calculate_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Calculate multichannel structural similarity for two uint8 images."""

    _validate_pair(reference, candidate)
    return float(
        structural_similarity(
            reference,
            candidate,
            channel_axis=2,
            data_range=255,
        )
    )


def variant_name(prefix: str, value: float) -> str:
    """Create stable readable names such as ``sigma30`` or ``alpha050``."""

    if value <= 0:
        raise ValueError("variant value must be positive")
    if value <= 1:
        formatted = f"{int(round(value * 100)):03d}"
    elif float(value).is_integer():
        formatted = str(int(value))
    else:
        formatted = str(value).replace(".", "p")
    return f"{prefix}{formatted}"
