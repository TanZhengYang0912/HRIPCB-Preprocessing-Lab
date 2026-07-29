import pytest

from hripcb_dashboard.filtering import (
    best_by_module,
    best_experiment,
    collapse_shared_baseline,
    comparison_records,
    filter_records,
    is_combined_record,
    inference_widget_keys,
    normalize_selection,
    option_values,
    reset_selection_state,
)


@pytest.fixture
def sample_records():
    return [
        {"id": "member1_gaussian", "model_id": "baseline", "module": "member1", "technique": "gaussian", "split": "val"},
        {"id": "member2_median", "model_id": "baseline", "module": "member2", "technique": "median", "split": "val"},
        {"id": "member2_clahe", "model_id": "baseline", "module": "member2", "technique": "clahe", "split": "val"},
        {"id": "member3_bilateral", "model_id": "baseline", "module": "member3", "technique": "bilateral", "split": "val"},
        {"id": "member4_nlm", "model_id": "final", "module": "member4", "technique": "nlm", "split": "test"},
    ]


def test_member2_options_only_include_member2_techniques(sample_records):
    options = option_values(sample_records, module="member2")
    assert options["technique"] == ["clahe", "median"]
    assert "gaussian" not in options["technique"]
    assert "bilateral" not in options["technique"]


def test_upstream_selection_cascades_model_split_module_and_technique(sample_records):
    options = option_values(sample_records, model="baseline", split="val", module="member2")
    assert options["model"] == ["baseline", "final"]
    assert options["split"] == ["val"]
    assert options["module"] == ["member1", "member2", "member3"]
    assert options["technique"] == ["clahe", "median"]


def test_invalid_downstream_selection_resets_to_all(sample_records):
    normalized = normalize_selection(
        sample_records,
        {"model": "baseline", "split": "val", "module": "member2", "technique": "gaussian"},
    )
    assert normalized == {"model": "baseline", "split": "val", "module": "member2", "technique": "all"}


def test_filter_records_applies_all_active_values(sample_records):
    filtered = filter_records(sample_records, model="baseline", split="val", module="member2", technique="median")
    assert [record["id"] for record in filtered] == ["member2_median"]


def test_best_experiment_uses_validation_ablation_and_map5095():
    records = [
        {"id": "val_best", "module": "member2", "technique": "median_clahe", "split": "val", "evaluation_type": "ablation", "metrics": {"map50_95": 0.8}},
        {"id": "test_higher", "module": "member2", "split": "test", "evaluation_type": "official_test", "metrics": {"map50_95": 0.99}},
        {"id": "final_higher", "module": "member4", "split": "val", "evaluation_type": "official_final", "metrics": {"map50_95": 0.95}},
    ]
    assert best_experiment(records)["id"] == "val_best"


def test_best_experiment_ignores_higher_scoring_single_technique():
    records = [
        {"id": "noise_only", "module": "member2", "technique": "median", "split": "val", "evaluation_type": "ablation", "metrics": {"map50_95": 0.95}},
        {"id": "combined", "module": "member2", "technique": "median_clahe", "split": "val", "evaluation_type": "ablation", "metrics": {"map50_95": 0.80}},
    ]

    assert is_combined_record(records[0]) is False
    assert is_combined_record(records[1]) is True
    assert best_experiment(records)["id"] == "combined"


def test_best_by_module_returns_one_recommendation_per_module():
    records = [
        {"id": "member1_low", "module": "member1", "technique": "gaussian_bbhe", "split": "val", "evaluation_type": "ablation", "metrics": {"map50_95": 0.4}},
        {"id": "member1_high", "module": "member1", "technique": "gaussian", "split": "val", "evaluation_type": "ablation", "metrics": {"map50_95": 0.6}},
        {"id": "member2_best", "module": "member2", "technique": "median_clahe", "split": "val", "evaluation_type": "ablation", "metrics": {"map50_95": 0.5}},
    ]
    assert {record["id"] for record in best_by_module(records)} == {"member1_low", "member2_best"}


def test_collapse_shared_baseline_replaces_duplicate_original_member_rows():
    records = [
        {"id": "original", "model_id": "baseline", "module": f"member{i}", "technique": "original", "split": "val"}
        for i in range(1, 5)
    ]
    records.append({"id": "combined", "model_id": "baseline", "module": "member1", "technique": "gaussian_bbhe", "split": "val"})

    collapsed = collapse_shared_baseline(records)

    controls = [record for record in collapsed if record["id"] == "original_shared_control"]
    assert len(controls) == 1
    assert controls[0]["module"] == "baseline"
    assert controls[0]["technique"] == "original"
    assert controls[0]["shared_control_modules"] == ["member1", "member2", "member3", "member4"]
    assert len(collapsed) == 2


def test_combined_comparison_includes_shared_original_but_not_single_techniques():
    records = [
        {"id": "original", "model_id": "baseline", "module": "member1", "technique": "original", "split": "val"},
        {"id": "noise_only", "model_id": "baseline", "module": "member1", "technique": "gaussian", "split": "val"},
        {"id": "combined", "model_id": "baseline", "module": "member1", "technique": "gaussian_bbhe", "split": "val"},
    ]

    visible = comparison_records(records, run_type="combined")

    assert {record["id"] for record in visible} == {"original", "combined"}


def test_reset_selection_state_resets_only_the_requested_filter_group():
    state = {
        "compare_model": "baseline",
        "compare_split": "val",
        "infer_model": "final",
    }
    reset_selection_state(state, prefix="compare_")
    assert state == {
        "compare_model": "all",
        "compare_split": "all",
        "infer_model": "final",
    }


def test_reset_selection_state_can_restore_combined_default():
    state = {"compare_model": "baseline", "compare_run_type": "reference"}

    reset_selection_state(
        state,
        prefix="compare_",
        extra_fields=("run_type",),
        defaults={"run_type": "combined"},
    )

    assert state == {"compare_model": "all", "compare_run_type": "combined"}


def test_inference_tabs_receive_distinct_widget_keys():
    assert inference_widget_keys("infer") == ("infer_model", "infer_module", "infer_technique")
    assert inference_widget_keys("video") == ("video_model", "video_module", "video_technique")
    assert set(inference_widget_keys("infer")).isdisjoint(inference_widget_keys("video"))
