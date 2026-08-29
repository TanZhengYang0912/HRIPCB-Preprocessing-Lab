import importlib


def _search_module():
    return importlib.import_module("scripts.run_member2_wavelet_search")


def test_top_wavelets_ignores_original_and_ranks_by_map5095():
    records = [
        {"id": "original", "technique": "original", "metrics": {"map50_95": 0.99}},
        {"id": "wavelet_db4", "technique": "wavelet", "metrics": {"map50_95": 0.52}},
        {"id": "wavelet_sym4", "technique": "wavelet", "metrics": {"map50_95": 0.54}},
        {"id": "wavelet_coif1", "technique": "wavelet", "metrics": {"map50_95": 0.51}},
    ]

    winners = _search_module().top_wavelets(records, count=2)

    assert [record["id"] for record in winners] == ["wavelet_sym4", "wavelet_db4"]


def test_refinement_presets_expand_two_winners_into_forty_candidates():
    winners = [
        {"parameters": {"wavelet_name": "sym4"}},
        {"parameters": {"wavelet_name": "db4"}},
    ]
    refinement = {
        "methods": ["BayesShrink", "VisuShrink"],
        "modes": ["soft", "hard"],
        "levels": [None, 1, 2, 3, 4],
    }

    presets = _search_module().build_refinement_presets(winners, refinement)

    assert len(presets) == 40
    assert len({preset["id"] for preset in presets}) == 40
    assert {preset["wavelet"] for preset in presets} == {"sym4", "db4"}
    assert {preset.get("wavelet_levels") for preset in presets} == {None, 1, 2, 3, 4}
