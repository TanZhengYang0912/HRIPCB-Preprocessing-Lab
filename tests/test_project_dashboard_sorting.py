import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_project_dashboard import aggregate_results

from hripcb_dashboard.dashboard import write_dashboard_html


def test_project_aggregate_does_not_resurrect_obsolete_candidate(tmp_path):
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
    candidate_dir = runs_root / "retrained_median_candidate" / "evaluation"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "results.json").write_text(json.dumps([{
        "id": "obsolete_candidate",
        "model_id": "retrained_median_candidate",
        "module": "member2",
        "technique": "median",
        "split": "test",
        "metrics": {"map50_95": 0.9},
    }]))

    aggregate_results(runs_root, tmp_path / "output")
    rows = json.loads((tmp_path / "output" / "results.json").read_text())

    assert [row["id"] for row in rows] == ["member1_original"]


def test_dashboard_contains_model_and_metric_sorting_controls(tmp_path):
    records = [
        {
            "id": "member2_median_k3",
            "module": "member2",
            "technique": "median",
            "model_id": "baseline",
            "model_label": "Baseline YOLO",
            "split": "val",
            "parameters": {"median_kernel_size": 3},
            "metrics": {"map50_95": 0.4, "f1": 0.7},
            "preview": "../member2_parameter_sweep/previews/member2_median_k3.jpg",
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
