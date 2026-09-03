from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.streamlit_dashboard as streamlit_dashboard
import hripcb_dashboard.video as video_dashboard
from hripcb_dashboard.analysis import build_analysis_payload, technique_label
from hripcb_dashboard.dashboard import write_dashboard_html
from hripcb_dashboard.filtering import (
    best_by_module,
    collapse_shared_baseline,
    comparison_records,
    is_combined_record,
    option_values,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_project_dashboard import aggregate_results, write_project_reports


def _member5_record(
    record_id: str,
    technique: str,
    *,
    score: float = 0.5,
    evaluation_type: str = "ablation",
    split: str = "val",
) -> dict:
    return {
        "id": record_id,
        "model_id": "baseline",
        "module": "member5",
        "technique": technique,
        "evaluation_type": evaluation_type,
        "split": split,
        "parameters": {
            "tv_weight": 0.02,
            "morphology_kernel_size": 9,
            "top_hat_amount": 1.0,
            "black_hat_amount": 0.5,
        },
        "metrics": {
            "map50_95": score,
            "map50": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
        },
    }


def test_member5_technique_labels_cover_controls_and_combined_pipeline():
    assert technique_label("tv") == "TV"
    assert technique_label("top_black_hat") == "Top-hat + Black-hat"
    assert technique_label("tv_top_black_hat") == "TV + Top-hat + Black-hat"


def test_member5_combined_filter_and_shared_original_collapsing():
    records = [
        {
            "id": f"original_{module}",
            "model_id": "baseline",
            "module": module,
            "technique": "original",
            "split": "val",
        }
        for module in ("member1", "member2", "member3", "member4", "member5")
    ]
    records.extend([
        _member5_record("member5_tv", "tv"),
        _member5_record("member5_morphology", "top_black_hat"),
        _member5_record("member5_combined", "tv_top_black_hat"),
    ])

    assert is_combined_record(records[-1])
    assert option_values(records, module="member5")["technique"] == [
        "original",
        "top_black_hat",
        "tv",
        "tv_top_black_hat",
    ]
    assert {row["id"] for row in comparison_records(records, run_type="combined")} == {
        "original_shared_control",
        "member5_combined",
    }
    control = next(row for row in collapse_shared_baseline(records) if row["id"] == "original_shared_control")
    assert control["shared_control_modules"] == [
        "member1",
        "member2",
        "member3",
        "member4",
        "member5",
    ]


def test_member5_is_included_in_best_by_module_and_analysis_stage_comparison():
    original = [
        {
            "id": f"original_{module}",
            "model_id": "baseline",
            "module": module,
            "technique": "original",
            "evaluation_type": "ablation",
            "split": "val",
            "metrics": {"map50_95": 0.4},
        }
        for module in ("member1", "member2", "member3", "member4", "member5")
    ]
    records = original + [
        _member5_record("member5_tv", "tv", score=0.45),
        _member5_record("member5_morphology", "top_black_hat", score=0.46),
        _member5_record("member5_combined", "tv_top_black_hat", score=0.6),
    ]

    winners = best_by_module(records)
    assert [row["id"] for row in winners] == ["member5_combined"]
    payload = build_analysis_payload(records)
    assert any(row["member"] == "member5" for row in payload["stage_comparison"])
    assert any(row["id"] == "member5_combined" for row in payload["combined_winners"])


def test_static_dashboard_contains_member5_combined_and_shared_control_logic(tmp_path):
    records = [
        _member5_record("member5_combined", "tv_top_black_hat"),
        {
            "id": "original_member5",
            "model_id": "baseline",
            "module": "member5",
            "technique": "original",
            "split": "val",
            "parameters": {},
            "metrics": {"map50_95": 0.4},
        },
    ]
    dashboard_path = write_dashboard_html(tmp_path, records)
    html = dashboard_path.read_text(encoding="utf-8")

    assert "tv_top_black_hat" in html
    assert "member5" in html
    assert "Top-hat + Black-hat" in html or "top_black_hat" in html
    assert "localeCompare" in html


def test_streamlit_member5_candidate_preserves_all_parameters_and_runs_image_pair(monkeypatch):
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    selected = _member5_record("member5_combined", "tv_top_black_hat")
    seen_candidates = []

    def fake_apply_candidate(input_image, candidate):
        seen_candidates.append(candidate)
        return input_image + 3

    monkeypatch.setattr(streamlit_dashboard, "apply_candidate", fake_apply_candidate)
    monkeypatch.setattr(
        streamlit_dashboard,
        "_detect",
        lambda model, input_image: (input_image.copy(), 1),
    )

    result = streamlit_dashboard._run_inference_pair(object(), image, selected)

    assert seen_candidates == [{
        "module": "member5",
        "technique": "tv_top_black_hat",
        "parameters": selected["parameters"],
    }]
    assert result["processed"].shape == image.shape
    assert result["preprocessed_detections"] == 1


def test_streamlit_member5_parameter_panel_names_all_controls():
    source = Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    assert "MEMBER5_PARAMETER_KEYS" in source
    for parameter in ("tv_weight", "morphology_kernel_size", "top_hat_amount", "black_hat_amount"):
        assert parameter in source
    assert "Member 1–5" in source


class _MetricColumn:
    def metric(self, *_args, **_kwargs):
        return None


class _ParameterPanelStub:
    def __init__(self):
        self.parameter_payload = None

    def subheader(self, *_args, **_kwargs):
        return None

    def caption(self, *_args, **_kwargs):
        return None

    def columns(self, count):
        return [_MetricColumn() for _ in range(count)]

    @contextmanager
    def expander(self, *_args, **_kwargs):
        yield self

    def json(self, payload):
        self.parameter_payload = payload


def test_member5_parameter_panel_displays_each_control():
    streamlit = _ParameterPanelStub()

    streamlit_dashboard._render_active_experiment(
        streamlit,
        _member5_record("member5_combined", "tv_top_black_hat"),
    )

    assert list(streamlit.parameter_payload)[:4] == [
        "tv_weight",
        "morphology_kernel_size",
        "top_hat_amount",
        "black_hat_amount",
    ]


def test_video_inference_forwards_member5_candidate_parameters(tmp_path, monkeypatch):
    input_path = tmp_path / "member5_input.avi"
    output_path = tmp_path / "member5_output.mp4"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (16, 12),
    )
    assert writer.isOpened()
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    for _ in range(2):
        writer.write(frame)
    writer.release()

    seen_candidates = []

    def fake_apply_candidate(image, candidate):
        seen_candidates.append(candidate)
        return image

    class Result:
        boxes = []

        def plot(self):
            return frame.copy()

    class Model:
        def predict(self, **_kwargs):
            return [Result()]

    monkeypatch.setattr(video_dashboard, "apply_candidate", fake_apply_candidate)
    selected = _member5_record("member5_combined", "tv_top_black_hat")
    summary = video_dashboard.process_video(
        input_path,
        output_path,
        Model(),
        streamlit_dashboard._candidate_from_record(selected),
        imgsz=1024,
        conf=0.25,
        iou=0.7,
        device="cpu",
    )

    assert summary["frames"] == 2
    assert seen_candidates == [
        streamlit_dashboard._candidate_from_record(selected),
        streamlit_dashboard._candidate_from_record(selected),
    ]


def test_project_aggregate_excludes_partial_member5_and_includes_complete_results(tmp_path):
    runs_root = tmp_path / "runs"
    output_root = tmp_path / "output"
    member5_dir = runs_root / "member5_full_search"
    member5_dir.mkdir(parents=True)
    (member5_dir / "results.json").write_text(
        json.dumps([_member5_record("member5_partial", "tv_top_black_hat", score=0.9)]),
        encoding="utf-8",
    )
    (member5_dir / "progress.json").write_text(
        json.dumps({
            "status": "running",
            "candidate_ids": ["member5_partial", "member5_missing"],
            "completed_ids": ["member5_partial"],
            "fingerprint": "test",
            "records": [],
        }),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        aggregate_results(runs_root, output_root)
    assert not output_root.joinpath("results.json").exists()

    (member5_dir / "results.json").write_text(
        json.dumps([
            _member5_record("member5_partial", "tv_top_black_hat", score=0.9),
            _member5_record("member5_missing", "tv_top_black_hat", score=0.8),
        ]),
        encoding="utf-8",
    )
    (member5_dir / "progress.json").write_text(
        json.dumps({
            "status": "complete",
            "candidate_ids": ["member5_partial", "member5_missing"],
            "completed_ids": ["member5_partial", "member5_missing"],
            "fingerprint": "test",
            "records": [],
        }),
        encoding="utf-8",
    )
    aggregate_results(runs_root, output_root)
    rows = json.loads(output_root.joinpath("results.json").read_text(encoding="utf-8"))
    assert {row["module"] for row in rows} == {"member5"}
    assert len(rows) == 2


def test_member5_completion_requires_nonempty_matching_ids_and_preserves_official_rows(tmp_path):
    runs_root = tmp_path / "runs"
    output_root = tmp_path / "output"
    member5_dir = runs_root / "member5_full_search"
    member5_dir.mkdir(parents=True)
    existing = [
        {"id": "old_member5", "module": "member5", "split": "val", "evaluation_type": "ablation", "technique": "tv_top_black_hat"},
        {"id": "member5_official", "module": "member5", "split": "test", "evaluation_type": "official_test", "technique": "tv_top_black_hat"},
    ]
    output_root.mkdir()
    (output_root / "results.json").write_text(json.dumps(existing), encoding="utf-8")
    (member5_dir / "results.json").write_text("[]", encoding="utf-8")
    (member5_dir / "progress.json").write_text(
        json.dumps({"status": "complete", "candidate_ids": [], "completed_ids": [], "records": [], "fingerprint": "test"}),
        encoding="utf-8",
    )
    aggregate_results(runs_root, output_root)
    assert json.loads((output_root / "results.json").read_text(encoding="utf-8")) == existing

    complete = _member5_record("new_member5", "tv_top_black_hat", score=0.8)
    (member5_dir / "results.json").write_text(json.dumps([complete]), encoding="utf-8")
    (member5_dir / "progress.json").write_text(
        json.dumps({"status": "complete", "candidate_ids": ["new_member5"], "completed_ids": ["new_member5"], "records": [], "fingerprint": "test"}),
        encoding="utf-8",
    )
    aggregate_results(runs_root, output_root)
    rows = json.loads((output_root / "results.json").read_text(encoding="utf-8"))
    assert {row["id"] for row in rows} == {"new_member5", "member5_official"}


def test_write_project_reports_refreshes_derived_files_without_touching_results(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    results_path = output_root / "results.json"
    results_path.write_text('{"sentinel": true}', encoding="utf-8")
    records = [_member5_record("member5_combined", "tv_top_black_hat", score=0.8)]

    write_project_reports(output_root, records, source_files=["member5/results.json"])

    assert results_path.read_text(encoding="utf-8") == '{"sentinel": true}'
    selection = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    assert selection["best_by_module"]["member5"]["id"] == "member5_combined"
    assert (output_root / "results.csv").is_file()
    assert (output_root / "dashboard.html").is_file()
