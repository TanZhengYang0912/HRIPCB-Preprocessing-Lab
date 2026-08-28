from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import cv2
import numpy as np
import pytest

from hripcb_dashboard.batch import extract_image_entries
from hripcb_dashboard import reporting
from hripcb_dashboard.reporting import build_report_pdf, format_parameters, record_metric_summary, report_chart_payload
from hripcb_dashboard.analysis import build_analysis_payload
from hripcb_dashboard.video import process_video
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_official_test_comparison import build as build_official_test_comparison


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_extract_image_entries_accepts_images_and_rejects_unsafe_or_unsupported_files():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    files, skipped = extract_image_entries(
        [
            ("batch.zip", _zip_bytes({"nested/one.jpg": encoded.tobytes(), "../escape.jpg": encoded.tobytes(), "notes.txt": b"no"})),
            ("single.png", encoded.tobytes()),
        ]
    )

    assert [name for name, _ in files] == ["nested/one.jpg", "single.png"]
    assert any("unsafe" in message.lower() for message in skipped)
    assert any("unsupported" in message.lower() for message in skipped)


def test_record_metric_summary_ranks_runs_and_calculates_coverage():
    records = [
        {"id": "a", "model_id": "baseline", "module": "member1", "metrics": {"map50_95": 0.40, "f1": 0.80}},
        {"id": "b", "model_id": "final", "module": "member2", "metrics": {"map50_95": 0.55, "f1": 0.85}},
    ]

    summary = record_metric_summary(records)

    assert summary["count"] == 2
    assert summary["module_count"] == 2
    assert summary["best"]["id"] == "b"
    assert summary["metric"] == "map50_95"


def test_record_metric_summary_excludes_baseline_control_from_module_count():
    records = [
        {"id": "member1", "model_id": "baseline", "module": "member1", "metrics": {"map50_95": 0.40}},
        {"id": "member2", "model_id": "baseline", "module": "member2", "metrics": {"map50_95": 0.41}},
        {"id": "control", "model_id": "baseline", "module": "baseline", "metrics": {"map50_95": 0.39}},
    ]

    summary = record_metric_summary(records)

    assert summary["module_count"] == 2
    assert summary["baseline_control_count"] == 1


def test_record_metric_summary_uses_combined_runs_for_best_run():
    records = [
        {"id": "noise_only", "module": "member1", "technique": "gaussian", "metrics": {"map50_95": 0.90}},
        {"id": "combined", "module": "member1", "technique": "gaussian_bbhe", "metrics": {"map50_95": 0.80}},
    ]

    summary = record_metric_summary(records)

    assert summary["best"]["id"] == "combined"
    assert summary["combined_count"] == 1


def test_record_metric_summary_exposes_consistent_display_and_reference_counts():
    records = [
        {
            "id": "original",
            "model_id": "baseline",
            "module": f"member{i}",
            "technique": "original",
            "split": "val",
            "metrics": {"map50_95": 0.5},
        }
        for i in range(1, 5)
    ]
    records.extend([
        {
            "id": f"combined_{i}",
            "model_id": "baseline",
            "module": f"member{i}",
            "technique": (
                "gaussian_bbhe" if i == 1 else
                "wavelet_homomorphic" if i == 2 else
                "bilateral_agcwd" if i == 3 else "nlm_msr"
            ),
            "split": "val",
            "metrics": {"map50_95": 0.51},
        }
        for i in range(1, 5)
    ])

    summary = record_metric_summary(records)

    assert summary["count"] == 8
    assert summary["display_count"] == 5
    assert summary["combined_count"] == 4
    assert summary["reference_count"] == 1


def test_analysis_payload_contains_original_and_four_combined_winners():
    records = [
        {
            "id": "original",
            "model_id": "baseline",
            "module": "member1",
            "technique": "original",
            "split": "val",
            "evaluation_type": "ablation",
            "metrics": {"map50_95": 0.5151, "map50": 0.942, "precision": 0.9719, "recall": 0.944, "f1": 0.9577},
        },
        {
            "id": "member1_combined",
            "model_id": "baseline",
            "module": "member1",
            "technique": "gaussian_bbhe",
            "split": "val",
            "evaluation_type": "ablation",
            "parameters": {"bbhe_strength": 0.25},
            "metrics": {"map50_95": 0.51, "map50": 0.92, "precision": 0.96, "recall": 0.91, "f1": 0.94},
        },
        {
            "id": "member1_combined_lower",
            "model_id": "baseline",
            "module": "member1",
            "technique": "gaussian_bbhe",
            "split": "val",
            "evaluation_type": "ablation",
            "metrics": {"map50_95": 0.49},
        },
    ]

    payload = build_analysis_payload(records)

    assert [row["label"] for row in payload["original_vs_combined"]] == [
        "Original",
        "member1 / Gaussian + BBHE",
    ]
    assert payload["combined_winners"][0]["id"] == "member1_combined"
    assert payload["combined_winners"][0]["map50_95"] == 0.51
    assert payload["metric_comparison"][0]["technique"] == "Gaussian + BBHE"


def test_build_report_pdf_contains_a_real_pdf_header():
    records = [{
        "id": "member2_wavelet_sym4",
        "model_id": "baseline",
        "module": "member2",
        "technique": "wavelet",
        "split": "val",
        "parameters": {"wavelet_name": "sym4", "wavelet_method": "BayesShrink"},
        "metrics": {"map50_95": 0.5269, "precision": 0.96, "recall": 0.95, "map50": 0.97, "f1": 0.95},
    }]

    payload = build_report_pdf(records, {"imgsz": 1024, "conf": 0.25, "iou": 0.70})

    assert payload.startswith(b"%PDF")
    assert len(payload) > 500


def test_report_chart_payload_contains_report_ready_comparisons():
    records = [{
        "id": "original",
        "model_id": "baseline",
        "module": "member1",
        "technique": "original",
        "split": "val",
        "evaluation_type": "ablation",
        "metrics": {"map50_95": 0.5151},
    }, {
        "id": "combined",
        "model_id": "baseline",
        "module": "member1",
        "technique": "gaussian_bbhe",
        "split": "val",
        "evaluation_type": "ablation",
        "metrics": {"map50_95": 0.5109},
    }]

    charts = report_chart_payload(records)

    assert len(charts["original_vs_combined"]) == 2
    assert len(charts["metric_comparison"]) == 1
    assert len(charts["stage_comparison"]) == 4


def test_official_comparison_can_exclude_unrequested_retrained_candidate(tmp_path):
    original_path = tmp_path / "original.json"
    preprocessed_path = tmp_path / "preprocessed.json"
    candidate_path = tmp_path / "candidate.json"
    original_path.write_text(json.dumps({
        "metrics/precision(B)": 0.95,
        "metrics/recall(B)": 0.92,
        "metrics/mAP50(B)": 0.92,
        "metrics/mAP50-95(B)": 0.49,
    }))
    preprocessed_path.write_text(json.dumps([{
        "id": "baseline_wavelet_homomorphic",
        "model_id": "baseline",
        "module": "member2",
        "technique": "wavelet_homomorphic",
    }]))
    candidate_path.write_text(json.dumps([{"id": "unrequested_candidate", "model_id": "retrained_candidate"}]))

    output = build_official_test_comparison(tmp_path / "official", original_path, preprocessed_path, None)
    records = json.loads(output.read_text())

    assert [record["id"] for record in records] == ["baseline_original_test", "baseline_wavelet_homomorphic"]


def test_format_parameters_is_readable_for_long_parameter_sets():
    text = format_parameters({
        "bilateral_diameter": 7,
        "bilateral_sigma_color": 50.0,
        "bilateral_sigma_space": 50.0,
        "nlm_processing_max_side": 768,
    })

    assert "bilateral_diameter: 7" in text
    assert "\n" in text
    assert "{\"bilateral" not in text


def test_dumps_json_converts_non_finite_numbers_to_json_null():
    assert hasattr(reporting, "dumps_json")
    payload = reporting.dumps_json({"mean_psnr": float("inf"), "mean_ssim": 1.0})

    assert "Infinity" not in payload
    assert json.loads(payload)["mean_psnr"] is None


def test_streamlit_exposes_extra_effort_sections_and_frozen_protocol():
    source = Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    for label in (
        "Analysis & reports",
        "Download complete PDF report",
        "hripcb_preprocessing_report_v2.pdf",
        "Reproducibility",
        "Download selected experiment config",
        "Upload PCB images or one ZIP folder",
        '"zip"',
        "Video processing",
        '_render_recommendation(st, records, key_prefix="video")',
        'key=f"use_recommended_{key_prefix}"',
        "Download annotated video",
        "imgsz",
        "conf",
        "iou",
        "Best combined run",
        'run_type_options = ["all", "combined", "reference"]',
    ):
        assert label in source


def test_streamlit_does_not_reference_removed_retrained_model():
    source = Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    assert 'record.get("model_id") == "final"' not in source


class _FakeBoxes:
    def __len__(self):
        return 1


class _FakeResult:
    boxes = _FakeBoxes()

    def plot(self):
        return np.zeros((24, 32, 3), dtype=np.uint8)


class _FakeModel:
    def predict(self, **kwargs):
        assert kwargs["imgsz"] == 1024
        return [_FakeResult()]


def test_process_video_writes_annotated_video_and_summary(tmp_path):
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.mp4"
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    summary = process_video(
        input_path,
        output_path,
        _FakeModel(),
        {"technique": "original", "parameters": {}},
        imgsz=1024,
        conf=0.25,
        iou=0.70,
        device="cpu",
    )

    assert summary["frames"] == 3
    assert summary["detections"] == 3
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_process_video_outputs_browser_compatible_h264_when_ffmpeg_is_available(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required for browser codec verification")

    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.mp4"
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    summary = process_video(
        input_path,
        output_path,
        _FakeModel(),
        {"technique": "original", "parameters": {}},
        imgsz=1024,
        conf=0.25,
        iou=0.70,
        device="cpu",
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert summary["browser_compatible"] is True
    assert probe.stdout.strip() == "h264"
