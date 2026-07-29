import numpy as np

from hripcb_member1.report import build_comparison_grid, write_comparison_html


def test_report_contains_all_required_panel_labels(tmp_path):
    html_path = write_comparison_html(
        tmp_path,
        {
            "source": "sample.jpg",
            "parameters": "kernel=5x5, sigmaX=1.0",
            "panels": [
                {"label": "Original", "src": "original.jpg"},
                {"label": "Gaussian Filtering", "src": "gaussian.jpg"},
                {"label": "BBHE", "src": "bbhe.jpg"},
                {"label": "Gaussian + BBHE", "src": "combined.jpg"},
            ],
            "model_metrics": [
                {
                    "variant": "original",
                    "precision": 0.95,
                    "recall": 0.92,
                    "map50": 0.91,
                    "map50_95": 0.49,
                    "f1": 0.93,
                },
                {
                    "variant": "gaussian",
                    "precision": 0.96,
                    "recall": 0.93,
                    "map50": 0.92,
                    "map50_95": 0.50,
                    "f1": 0.94,
                },
                {
                    "variant": "bbhe",
                    "precision": 0.79,
                    "recall": 0.58,
                    "map50": 0.54,
                    "map50_95": 0.27,
                    "f1": 0.67,
                },
                {
                    "variant": "gaussian_bbhe",
                    "precision": 0.90,
                    "recall": 0.64,
                    "map50": 0.70,
                    "map50_95": 0.36,
                    "f1": 0.75,
                },
            ],
        },
    )
    html = html_path.read_text()
    assert "Gaussian Filtering" in html
    assert "BBHE" in html
    assert "Gaussian + BBHE" in html
    assert "mAP50" in html
    assert "0.9100" in html
    assert html.count("YOLO scores") == 4
    assert "0.9600" in html
    assert "0.7000" in html
    assert "Noisy" not in html
    assert "Low Contrast" not in html
    assert "sample.jpg" in html


def test_comparison_grid_is_written(tmp_path):
    images = {name: np.zeros((30, 40, 3), dtype=np.uint8) for name in ("one", "two")}
    output = build_comparison_grid(images, tmp_path / "grid.jpg")
    assert output.is_file()
