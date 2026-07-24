"""Experiment configuration and selection helpers for Member 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BilateralConfig:
    """One edge-preserving filter configuration."""

    diameter: int
    sigma_color: float
    sigma_space: float

    @property
    def tag(self) -> str:
        return (
            f"d{self.diameter}_c{self.sigma_color:g}_s{self.sigma_space:g}"
        )


NOISE_SIGMAS: tuple[int, ...] = (10, 25, 40)
NOISE_SEEDS: dict[int, int] = {10: 42, 25: 43, 40: 44}
AGCWD_ALPHAS: tuple[float, ...] = (0.5, 0.75, 1.0)
BILATERAL_CANDIDATES: tuple[BilateralConfig, ...] = (
    BilateralConfig(3, 25.0, 25.0),
    BilateralConfig(5, 50.0, 50.0),
    BilateralConfig(7, 75.0, 75.0),
)


def build_condition_names(noise_sigmas: Sequence[int]) -> list[str]:
    """Return the ordered clean/noisy/ablation condition names."""

    names = ["clean"]
    for sigma in noise_sigmas:
        names.extend(
            [
                f"noisy_sigma{sigma}",
                f"bilateral_sigma{sigma}",
                f"agcwd_sigma{sigma}",
                f"member3_sigma{sigma}",
            ]
        )
    return names


def choose_best_config(
    scores: Mapping[BilateralConfig, Mapping[int, float]],
) -> BilateralConfig:
    """Choose the config with the highest mean validation mAP50-95."""

    if not scores:
        raise ValueError("scores must not be empty")

    return max(scores, key=lambda config: _mean_score(scores[config]))


def choose_best_member3_parameters(
    scores: Mapping[tuple[BilateralConfig, float], Mapping[int, float]],
) -> tuple[BilateralConfig, float]:
    """Choose the Bilateral/AGCWD pair with the highest mean validation score."""

    if not scores:
        raise ValueError("scores must not be empty")
    return max(scores, key=lambda config: _mean_score(scores[config]))


def _mean_score(values: Mapping[int, float]) -> float:
    if not values:
        raise ValueError("scores for each config must not be empty")
    return sum(float(value) for value in values.values()) / len(values)
