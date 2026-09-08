from hripcb_dashboard.analysis import (
    EXTRA_STUDY_MODULES,
    MODULE_DISPLAY,
    is_extra_study,
    module_label,
)


def test_module_label_maps_every_member_key_to_its_assigned_name():
    assert module_label("member4") == "Joshua Lau Hao Jie"
    assert module_label("member5") == "Ng Chi Hao"
    assert module_label("member1") == "Tan Chun Jie"
    assert module_label("member3") == "Tan Zheng Yang"


def test_module_label_marks_member2_as_an_extra_study():
    assert module_label("member2") == "Extra study"
    assert is_extra_study("member2") is True
    assert EXTRA_STUDY_MODULES == frozenset({"member2"})


def test_module_label_names_the_baseline_control():
    assert module_label("baseline") == "Baseline control"
    assert is_extra_study("baseline") is False


def test_module_label_returns_unknown_keys_unchanged():
    assert module_label("member9") == "member9"
    assert module_label("") == "unknown"
    assert module_label(None) == "unknown"


def test_module_display_covers_every_assigned_member_and_nothing_else():
    assert set(MODULE_DISPLAY) == {
        "member1",
        "member2",
        "member3",
        "member4",
        "member5",
        "baseline",
    }


from pathlib import Path

from hripcb_dashboard.analysis import build_analysis_payload


def _record(module, technique, map50_95, record_id):
    return {
        "id": record_id,
        "module": module,
        "technique": technique,
        "split": "val",
        "model_id": "baseline",
        "evaluation_type": "ablation",
        "parameters": {},
        "metrics": {
            "precision": 0.9,
            "recall": 0.9,
            "map50": 0.9,
            "map50_95": map50_95,
            "f1": 0.9,
        },
    }


def test_analysis_payload_labels_use_member_names():
    records = [
        _record("member5", "original", 0.51, "original"),
        _record("member5", "tv_top_black_hat", 0.52, "tv_combo"),
        _record("member4", "nlm_msr", 0.22, "nlm_combo"),
    ]

    payload = build_analysis_payload(records)
    labels = [row["label"] for row in payload["original_vs_combined"]]

    assert any(label.startswith("Ng Chi Hao / ") for label in labels)
    assert any(label.startswith("Joshua Lau Hao Jie / ") for label in labels)
    assert not any("member" in label for label in labels)


def test_streamlit_dashboard_displays_modules_through_module_label():
    source = Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    assert "module_label" in source
    # No display string may interpolate the raw module key any more.
    assert "{best.get('module', '—')} /" not in source
    assert "{record.get('module', '—')} ·" not in source
    assert "{recommended.get('module', '—')} /" not in source


from hripcb_dashboard.reporting import record_metric_summary


def test_summary_excludes_extra_study_from_the_member_count():
    records = [
        _record("member1", "gaussian_bbhe", 0.51, "m1"),
        _record("member3", "bilateral_agcwd", 0.45, "m3"),
        _record("member4", "nlm_msr", 0.22, "m4"),
        _record("member5", "tv_top_black_hat", 0.52, "m5"),
        _record("member2", "wavelet_homomorphic", 0.51, "m2"),
    ]

    summary = record_metric_summary(records)

    assert summary["module_count"] == 4
    assert summary["extra_study_count"] == 1


def test_streamlit_dashboard_no_longer_claims_five_combined_techniques():
    source = Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    assert "five combined" not in source
    assert "Five combined" not in source
    assert "five member combined" not in source
    assert "All five winners" not in source


from hripcb_dashboard.analysis import ranking_chart_rows


def test_ranking_chart_rows_sorts_by_metric_and_returns_the_baseline():
    records = [
        _record("member5", "original", 0.5151, "original"),
        _record("member4", "nlm_msr", 0.2248, "m4_combo"),
        _record("member5", "tv_top_black_hat", 0.5239, "m5_combo"),
    ]

    rows, baseline = ranking_chart_rows(records)

    assert baseline == 0.5151
    assert [row["value"] for row in rows] == [0.5239, 0.2248]
    assert rows[0]["member"] == "Ng Chi Hao"
    assert rows[0]["label"] == "Ng Chi Hao / TV + Top-hat + Black-hat"
    assert rows[1]["member"] == "Joshua Lau Hao Jie"


def test_ranking_chart_rows_flags_the_extra_study():
    records = [
        _record("member2", "wavelet_homomorphic", 0.5171, "m2_combo"),
        _record("member5", "tv_top_black_hat", 0.5239, "m5_combo"),
    ]

    rows, baseline = ranking_chart_rows(records)

    assert baseline is None
    assert [row["is_extra"] for row in rows] == [False, True]


def test_ranking_chart_rows_honours_a_different_metric():
    records = [
        _record("member5", "original", 0.5151, "original"),
        _record("member5", "tv", 0.5281, "m5_tv"),
    ]
    records[1]["metrics"]["recall"] = 0.9662

    rows, baseline = ranking_chart_rows(records, metric="recall")

    assert baseline == 0.9
    assert rows[0]["value"] == 0.9662


def test_streamlit_dashboard_uses_left_sidebar_navigation():
    source = Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    # Five pages, reached via a left sidebar rather than a top segmented control.
    assert 'NAV_DASHBOARD = "Dashboard"' in source
    assert 'NAV_EXPERIMENTS = "Experiments"' in source
    assert 'NAV_IMAGE_INFERENCE = "Image processing"' in source
    assert 'NAV_ANALYSIS = "Analysis & reports"' in source
    assert 'NAV_VIDEO = "Video processing"' in source
    assert "def _render_sidebar_nav(" in source
    assert "with st.sidebar:" in source
    # The old top segmented-control mode toggle is gone.
    assert "st.segmented_control(" not in source
    assert 'st.tabs(["Run image inference", "Video processing"])' not in source
    assert 'st.tabs(["Compare experiments", "Analysis & reports"])' not in source


def test_streamlit_dashboard_uses_no_emoji_in_nav_page_labels():
    source = Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    start = source.index("NAV_DASHBOARD =")
    end = source.index("NAV_STATE_KEY")
    nav_labels_block = source[start:end]

    assert all(ord(character) < 128 for character in nav_labels_block)
    # The nav buttons render the bare page name -- no icon dict, no emoji prefix.
    assert "NAV_ICONS" not in source
    assert 'st.button(\n                    page,' in source


def test_ranking_chart_rows_keeps_only_the_best_run_per_technique():
    # A parameter sweep produces many runs of the same module+technique. The
    # chart must collapse them to one bar each, or it grows to hundreds of rows
    # with duplicate labels that Vega then drops.
    records = [
        _record("member5", "tv_top_black_hat", 0.4801, "sweep_w1"),
        _record("member5", "tv_top_black_hat", 0.5239, "sweep_w2"),
        _record("member5", "tv_top_black_hat", 0.5102, "sweep_w3"),
        _record("member4", "nlm_msr", 0.2248, "m4_combo"),
    ]

    rows, _ = ranking_chart_rows(records)

    assert [row["label"] for row in rows] == [
        "Ng Chi Hao / TV + Top-hat + Black-hat",
        "Joshua Lau Hao Jie / NLM + MSR",
    ]
    assert rows[0]["value"] == 0.5239


def test_ranking_chart_row_count_matches_distinct_labels():
    records = [
        _record("member5", "tv", 0.50 + index / 1000, f"tv_{index}")
        for index in range(20)
    ] + [_record("member1", "gaussian_bbhe", 0.44, "g1")]

    rows, _ = ranking_chart_rows(records)

    assert len(rows) == 2


def test_ranking_chart_is_sized_per_row_not_by_fixed_bar_height():
    source = Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    # A pinned mark height leaves thin bars floating in stretched bands
    # whenever the chart is resized (fullscreen, or few rows).
    assert "height=17" not in source
    assert "alt.Step(" in source


import re

# Matches a `.get("module"...)`/`.get('module'...)` call sitting inside an
# f-string interpolation `{...}` that is not already wrapped in module_label(...).
_RAW_MODULE_PATTERN = re.compile(r"""\{[^{}]*\.get\(\s*["']module["']""")
_RAW_TECHNIQUE_PATTERN = re.compile(r"""\{[^{}]*\.get\(\s*["']technique["']""")


def test_no_raw_module_or_technique_interpolation_in_dashboard_or_reporting():
    for path in (Path("scripts/streamlit_dashboard.py"), Path("src/hripcb_dashboard/reporting.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "f\"" not in line and "f'" not in line:
                continue
            if _RAW_MODULE_PATTERN.search(line) and "module_label(" not in line:
                raise AssertionError(f"{path}:{lineno}: raw module interpolation without module_label(): {line.strip()}")
            if _RAW_TECHNIQUE_PATTERN.search(line) and "technique_label(" not in line:
                raise AssertionError(f"{path}:{lineno}: raw technique interpolation without technique_label(): {line.strip()}")
