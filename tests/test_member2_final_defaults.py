import json
from pathlib import Path

import yaml

from scripts.evaluate_final_model import parse_args
from hripcb_preprocessing.candidates import build_candidates


EXPECTED_PARAMETERS = {
    "wavelet_name": "coif2",
    "wavelet_method": "VisuShrink",
    "wavelet_mode": "soft",
    "wavelet_levels": None,
    "homomorphic_gamma_low": 0.7,
    "homomorphic_gamma_high": 1.3,
    "homomorphic_cutoff": 20.0,
    "homomorphic_sharpness": 2.0,
}


def test_member2_sweep_contains_only_scanned_winner_presets():
    config = yaml.safe_load(Path("configs/member2_sweep.yaml").read_text())

    candidates = build_candidates("member2", config)
    combined = [row for row in candidates if row["technique"] == "wavelet_homomorphic"]

    assert len(combined) == 1
    assert combined[0]["parameters"] == EXPECTED_PARAMETERS


def test_final_evaluation_defaults_match_scanned_combined_winner():
    args = parse_args([])

    assert args.wavelet_name == "coif2"
    assert args.wavelet_method == "VisuShrink"
    assert args.wavelet_mode == "soft"
    assert args.wavelet_levels is None
    assert args.homomorphic_gamma_low == 0.7
    assert args.homomorphic_gamma_high == 1.3
    assert args.homomorphic_cutoff == 20.0
    assert args.homomorphic_sharpness == 2.0


def test_deployed_results_expose_scanned_combined_winner():
    records = json.loads(Path("runs/project_validation_comparison/results.json").read_text())
    winner = max(
        (
            row for row in records
            if row.get("module") == "member2"
            and row.get("technique") == "wavelet_homomorphic"
        ),
        key=lambda row: row["metrics"]["map50_95"],
    )

    assert winner["parameters"] == EXPECTED_PARAMETERS
    assert winner["metrics"]["map50_95"] == 0.5170690040261036
