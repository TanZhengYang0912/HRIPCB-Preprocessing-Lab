import subprocess
import sys


def test_member1_cli_is_runnable_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/run_member1_comparison.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout
