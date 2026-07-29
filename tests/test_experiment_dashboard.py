from hripcb_dashboard.dashboard import write_dashboard_html


def test_dashboard_renders_dynamic_module_parameters_and_metrics(tmp_path):
    records = [
        {
            "id": "member1_gaussian",
            "module": "member1",
            "technique": "gaussian",
            "parameters": {"gaussian_kernel_size": 7, "gaussian_sigma_x": 1.5},
            "metrics": {"map50_95": 0.4991, "recall": 0.9323},
            "preview": "previews/member1_gaussian.jpg",
        },
        {
            "id": "member2_bilateral",
            "module": "member2",
            "technique": "bilateral_agcwd",
            "parameters": {"diameter": 5, "sigma_color": 50, "gamma": 0.8},
            "metrics": {"map50_95": 0.512, "recall": 0.94},
            "preview": "previews/member2_bilateral.jpg",
        },
    ]

    path = write_dashboard_html(
        tmp_path,
        records,
        title="Preprocessing Parameter Sweep",
        primary_metric="map50_95",
    )

    html = path.read_text(encoding="utf-8")
    assert "gaussian_kernel_size" in html
    assert "sigma_color" in html
    assert "bilateral_agcwd" in html
    assert "mAP50-95" in html
    assert "member2_bilateral.jpg" in html
    assert "filterRows" in html
    assert "The primary metric is <strong>mAP50-95</strong>" in html
    assert "color-scheme: light" in html
    assert "--canvas: #f4f7fb" in html
    assert "prefers-reduced-transparency" in html
