import subprocess
import sys
from pathlib import Path


def test_evaluation_requires_a_checkpoint(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_baseline.py",
            "--weights",
            str(tmp_path / "missing.pt"),
            "--split",
            "test",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "checkpoint" in result.stderr.lower()
