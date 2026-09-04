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
    project_root = tmp_path / "project"
    relative_path = Path("runs/project_validation_comparison/results.json")
    results_path = project_root / relative_path
    results_path.parent.mkdir(parents=True)
    results_path.write_text('[{"id": "project-result"}]', encoding="utf-8")
    other_directory = tmp_path / "elsewhere"
    other_results = other_directory / relative_path
    other_results.parent.mkdir(parents=True)
    other_results.write_text('[{"id": "wrong-result"}]', encoding="utf-8")
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(other_directory)

    resolved = dashboard._resolve_results_path(relative_path)

    assert resolved == results_path
    assert dashboard._load_records(resolved) == [{"id": "project-result"}]
