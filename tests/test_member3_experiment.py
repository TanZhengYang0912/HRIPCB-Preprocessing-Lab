import pytest

from hripcb_baseline.member3_experiment import (
    BILATERAL_CANDIDATES,
    NOISE_SEEDS,
    build_condition_names,
    choose_best_config,
    choose_best_member3_parameters,
)


def test_noise_seeds_are_fixed_for_the_three_noise_levels():
    assert NOISE_SEEDS == {10: 42, 25: 43, 40: 44}


def test_condition_names_cover_the_required_ablation_matrix():
    names = build_condition_names((10, 25, 40))

    assert names == [
        "clean",
        "noisy_sigma10",
        "bilateral_sigma10",
        "agcwd_sigma10",
        "member3_sigma10",
        "noisy_sigma25",
        "bilateral_sigma25",
        "agcwd_sigma25",
        "member3_sigma25",
        "noisy_sigma40",
        "bilateral_sigma40",
        "agcwd_sigma40",
        "member3_sigma40",
    ]


def test_choose_best_config_uses_average_validation_map():
    scores = {
        BILATERAL_CANDIDATES[0]: {10: 0.40, 25: 0.50, 40: 0.60},
        BILATERAL_CANDIDATES[1]: {10: 0.55, 25: 0.52, 40: 0.51},
        BILATERAL_CANDIDATES[2]: {10: 0.35, 25: 0.55, 40: 0.65},
    }

    best = choose_best_config(scores)

    assert best == BILATERAL_CANDIDATES[1]


def test_choose_best_config_rejects_empty_scores():
    with pytest.raises(ValueError, match="scores"):
        choose_best_config({})


def test_choose_best_member3_parameters_uses_global_mean_score():
    scores = {
        (BILATERAL_CANDIDATES[0], 0.5): {10: 0.40, 25: 0.50, 40: 0.60},
        (BILATERAL_CANDIDATES[1], 0.75): {10: 0.55, 25: 0.52, 40: 0.51},
        (BILATERAL_CANDIDATES[2], 1.0): {10: 0.35, 25: 0.55, 40: 0.65},
    }

    best = choose_best_member3_parameters(scores)

    assert best == (BILATERAL_CANDIDATES[1], 0.75)
