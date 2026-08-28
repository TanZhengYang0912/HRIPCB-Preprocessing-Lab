from pathlib import Path

import scripts.streamlit_dashboard as dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_cloud_has_one_complete_requirements_manifest():
    manifests = sorted(PROJECT_ROOT.glob("requirements*.txt"))

    assert [path.name for path in manifests] == ["requirements.txt"]

    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for dependency in (
        "streamlit>=",
        "pandas>=",
        "numpy>=",
        "opencv-python-headless>=",
        "ultralytics-opencv-headless>=",
    ):
        assert dependency in requirements


def test_default_results_path_is_project_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    resolved = dashboard._resolve_results_path(Path("runs/project_validation_comparison/results.json"))

    assert resolved == dashboard.PROJECT_ROOT / "runs/project_validation_comparison/results.json"
    assert len(dashboard._load_records(resolved)) == 69
