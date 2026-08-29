import importlib


def _search_module():
    return importlib.import_module("scripts.run_member2_full_search")


def test_homomorphic_grid_expands_to_36_unique_presets():
    grid = {
        "gain_pairs": [[0.3, 1.8], [0.5, 1.5], [0.7, 1.3]],
        "cutoffs": [20, 40, 60, 80],
        "sharpness": [0.5, 1.0, 2.0],
    }

    presets = _search_module().build_homomorphic_presets(grid)

    assert len(presets) == 36
    assert len({preset["id"] for preset in presets}) == 36
    assert {preset["sharpness"] for preset in presets} == {0.5, 1.0, 2.0}


def test_best_by_technique_uses_exact_technique_and_map5095():
    records = [
        {"id": "original", "technique": "original", "metrics": {"map50_95": 0.99}},
        {"id": "h1", "technique": "homomorphic", "metrics": {"map50_95": 0.51}},
        {"id": "h2", "technique": "homomorphic", "metrics": {"map50_95": 0.53}},
        {"id": "combo", "technique": "wavelet_homomorphic", "metrics": {"map50_95": 0.52}},
    ]

    assert _search_module().best_by_technique(records, "homomorphic")["id"] == "h2"
    assert _search_module().best_by_technique(records, "wavelet_homomorphic")["id"] == "combo"


def test_partition_presets_splits_36_candidates_into_four_equal_shards():
    presets = [{"id": str(index)} for index in range(36)]

    shards = _search_module().partition_presets(presets, 4)

    assert [len(shard) for shard in shards] == [9, 9, 9, 9]
    assert [preset["id"] for shard in shards for preset in shard] == [
        str(index) for index in range(36)
    ]


def test_merge_shard_records_deduplicates_shared_candidates_and_fixes_previews():
    shards = [
        ("part_01", [
            {"id": "original", "preview": "previews/original.jpg"},
            {"id": "h1", "preview": "previews/h1.jpg"},
        ]),
        ("part_02", [
            {"id": "original", "preview": "previews/original.jpg"},
            {"id": "h2", "preview": "previews/h2.jpg"},
        ]),
    ]

    records = _search_module().merge_shard_records(shards)

    assert [record["id"] for record in records] == ["original", "h1", "h2"]
    assert [record["preview"] for record in records] == [
        "shards/part_01/previews/original.jpg",
        "shards/part_01/previews/h1.jpg",
        "shards/part_02/previews/h2.jpg",
    ]
