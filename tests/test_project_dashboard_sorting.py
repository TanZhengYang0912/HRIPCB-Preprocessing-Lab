import json

from hripcb_dashboard.dashboard import write_dashboard_html


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
    assert "syncCascadedFilters" in html
    assert "optionValuesFor" in html
    assert "Active experiment" in html
    assert "imgsz 1024" in html
    assert "conf 0.25" in html
    assert "IoU 0.70" in html
    assert json.dumps("Baseline YOLO") in html
