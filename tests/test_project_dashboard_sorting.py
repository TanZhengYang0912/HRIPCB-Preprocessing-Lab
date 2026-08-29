import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_project_dashboard import aggregate_results

from hripcb_dashboard.dashboard import write_dashboard_html


def test_project_aggregate_does_not_resurrect_retired_member2_results(tmp_path):
    runs_root = tmp_path / "runs"
    member_dir = runs_root / "member1_validation_sweep"
    member_dir.mkdir(parents=True)
    (member_dir / "results.json").write_text(json.dumps([{
        "id": "member1_original",
        "model_id": "baseline",
        "module": "member1",
        "technique": "original",
        "split": "val",
        "metrics": {"map50_95": 0.5},
    }]))
    retired_dir = runs_root / "baseline_median_test"
    retired_dir.mkdir(parents=True)
    (retired_dir / "results.json").write_text(json.dumps([{
        "id": "retired_member2_result",
        "model_id": "baseline",
        "module": "member2",
        "technique": "median",
        "split": "test",
        "metrics": {"map50_95": 0.9},
    }]))

    aggregate_results(runs_root, tmp_path / "output")
    rows = json.loads((tmp_path / "output" / "results.json").read_text())

    assert [row["id"] for row in rows] == ["member1_original"]


def test_project_aggregate_prefers_member2_full_search_results(tmp_path):
    runs_root = tmp_path / "runs"
    old_dir = runs_root / "member2_validation_sweep"
    full_dir = runs_root / "member2_full_search"
    old_dir.mkdir(parents=True)
    full_dir.mkdir(parents=True)
    (old_dir / "results.json").write_text(json.dumps([{
        "id": "old_member2_combined",
        "module": "member2",
        "technique": "wavelet_homomorphic",
        "split": "val",
        "evaluation_type": "ablation",
        "metrics": {"map50_95": 0.48},
    }]))
    (full_dir / "results.json").write_text(json.dumps([{
        "id": "scanned_member2_combined",
        "module": "member2",
        "technique": "wavelet_homomorphic",
        "split": "val",
        "evaluation_type": "ablation",
        "parameters": {
            "wavelet_name": "coif2",
            "homomorphic_cutoff": 20.0,
        },
        "metrics": {"map50_95": 0.5170690040261036},
    }]))

    aggregate_results(runs_root, tmp_path / "output")
    rows = json.loads((tmp_path / "output" / "results.json").read_text())

    assert [row["id"] for row in rows] == ["scanned_member2_combined"]


def test_project_aggregate_preserves_existing_modules_without_local_sweeps(tmp_path):
    runs_root = tmp_path / "runs"
    full_dir = runs_root / "member2_full_search"
    output_dir = tmp_path / "output"
    full_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (full_dir / "results.json").write_text(json.dumps([{
        "id": "original",
        "module": "member2",
        "technique": "wavelet_homomorphic",
        "split": "val",
        "evaluation_type": "ablation",
        "metrics": {"map50_95": 0.517},
    }]))
    (output_dir / "results.json").write_text(json.dumps([{
        "id": "original",
        "module": "member1",
        "technique": "gaussian_bbhe",
        "split": "val",
        "evaluation_type": "ablation",
        "metrics": {"map50_95": 0.51},
    }, {
        "id": "old_member2",
        "module": "member2",
        "technique": "wavelet_homomorphic",
        "split": "val",
        "evaluation_type": "ablation",
        "metrics": {"map50_95": 0.48},
    }]))

    aggregate_results(runs_root, output_dir)
    rows = json.loads((output_dir / "results.json").read_text())

    assert {(row["module"], row["id"]) for row in rows} == {
        ("member1", "original"),
        ("member2", "original"),
    }


def test_dashboard_contains_model_and_metric_sorting_controls(tmp_path):
    records = [
        {
            "id": "member2_wavelet_sym4",
            "module": "member2",
            "technique": "wavelet",
            "model_id": "baseline",
            "model_label": "Baseline YOLO",
            "split": "val",
            "parameters": {"wavelet_name": "sym4", "wavelet_method": "BayesShrink"},
            "metrics": {"map50_95": 0.4, "f1": 0.7},
            "preview": "../member2_validation_sweep/previews/wavelet_w_sym4.jpg",
        }
    ]
    path = write_dashboard_html(tmp_path, records, title="Project Sweep")
    html = path.read_text(encoding="utf-8")
    assert "id=\"modelFilter\"" in html
    assert "id=\"sortMetric\"" in html
    assert "id=\"sortDirection\"" in html
    assert "model_id" in html
    assert "split" in html
    assert "sortRows" in html
    assert "runTypeFilter" in html
    assert "runType: 'all'" in html
    assert "syncCascadedFilters" in html
    assert "optionValuesFor" in html
    assert "Active experiment" in html
    assert "imgsz 1024" in html
    assert "conf 0.25" in html
    assert "IoU 0.70" in html
    assert json.dumps("Baseline YOLO") in html
