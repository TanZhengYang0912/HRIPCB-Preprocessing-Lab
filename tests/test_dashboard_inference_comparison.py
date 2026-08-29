from pathlib import Path

from hripcb_dashboard.analysis import detection_comparison_row


def test_row_reports_both_counts_and_signed_change():
    row = detection_comparison_row("board.jpg", 5, 7, "baseline", "wavelet_w_sym4")

    assert row == {
        "file": "board.jpg",
        "original": 5,
        "processed": 7,
        "change": "+2",
        "model": "baseline",
        "experiment": "wavelet_w_sym4",
    }


def test_row_marks_a_drop_with_a_minus_sign():
    row = detection_comparison_row("board.jpg", 6, 4, "baseline", "homomorphic_h_c30")

    assert row["change"] == "-2"


def test_row_reports_no_change_without_a_sign():
    row = detection_comparison_row("board.jpg", 3, 3, "baseline", "original")

    assert row["change"] == "0"


def test_inference_mode_detects_on_original_and_processed():
    source = Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    for fragment in (
        "detection_comparison_row",
        "original_plotted, original_count = _detect(model, original)",
        "processed_plotted, processed_count = _detect(model, processed)",
        "Detection on original",
        "Detection after preprocessing",
    ):
        assert fragment in source
