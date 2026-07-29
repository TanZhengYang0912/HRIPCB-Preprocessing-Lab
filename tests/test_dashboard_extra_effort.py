from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import cv2
import numpy as np
import pytest

from hripcb_dashboard.batch import extract_image_entries
from hripcb_dashboard import reporting
from hripcb_dashboard.reporting import build_report_pdf, format_parameters, record_metric_summary
from hripcb_dashboard.video import process_video


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


def test_build_report_pdf_contains_a_real_pdf_header():
    records = [{
        "id": "member2_median_k5",
        "model_id": "baseline",
        "module": "member2",
        "technique": "median",
        "split": "val",
        "parameters": {"median_kernel_size": 5},
        "metrics": {"map50_95": 0.5269, "precision": 0.96, "recall": 0.95, "map50": 0.97, "f1": 0.95},
    }]

    payload = build_report_pdf(records, {"imgsz": 1024, "conf": 0.25, "iou": 0.70})

    assert payload.startswith(b"%PDF")
    assert len(payload) > 500


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
    ):
        assert label in source


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
