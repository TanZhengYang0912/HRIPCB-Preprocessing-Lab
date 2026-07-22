import numpy as np

from hripcb_member1.report import build_comparison_grid, write_comparison_html


def test_report_contains_all_required_panel_labels(tmp_path):
    html_path = write_comparison_html(
        tmp_path,
        {
            "source": "sample.jpg",
            "parameters": "sigma=30, alpha=0.50",
            "panels": [
                {"label": "Original", "src": "original.jpg"},
                {"label": "Noisy", "src": "noisy.jpg"},
                {"label": "Gaussian Filtering", "src": "gaussian.jpg"},
                {"label": "Low Contrast", "src": "low.jpg"},
                {"label": "BBHE", "src": "bbhe.jpg"},
                {"label": "Gaussian + BBHE", "src": "combined.jpg"},
            ],
        },
    )
    html = html_path.read_text()
    assert "Gaussian Filtering" in html
    assert "BBHE" in html
    assert "Gaussian + BBHE" in html
    assert "sample.jpg" in html


def test_comparison_grid_is_written(tmp_path):
    images = {name: np.zeros((30, 40, 3), dtype=np.uint8) for name in ("one", "two")}
    output = build_comparison_grid(images, tmp_path / "grid.jpg")
    assert output.is_file()
