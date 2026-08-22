from pathlib import Path


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
