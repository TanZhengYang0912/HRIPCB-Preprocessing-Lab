import numpy as np
import pytest

from hripcb_member1.metrics import (
    calculate_psnr,
    calculate_ssim,
    derive_variant_seed,
    variant_name,
)


def test_identical_images_have_infinite_psnr_and_unit_ssim():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert calculate_psnr(image, image) == float("inf")
    assert calculate_ssim(image, image) == pytest.approx(1.0)


def test_variant_seed_and_names_are_stable():
    first = derive_variant_seed(42, "01_missing_hole_06.jpg", "sigma30")
    second = derive_variant_seed(42, "01_missing_hole_06.jpg", "sigma30")
    assert first == second
    assert first != derive_variant_seed(42, "01_missing_hole_06.jpg", "sigma50")
    assert variant_name("sigma", 30) == "sigma30"
    assert variant_name("alpha", 0.5) == "alpha050"


def test_metrics_reject_mismatched_images():
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    second = np.zeros((8, 9, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="same shape"):
        calculate_psnr(first, second)
    with pytest.raises(ValueError, match="same shape"):
        calculate_ssim(first, second)
