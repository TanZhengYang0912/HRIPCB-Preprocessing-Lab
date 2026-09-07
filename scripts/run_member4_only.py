#!/usr/bin/env python3
"""Run just the Member 4 validation sweep (NLM + MSR) with the fixed MSR filter."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_preprocessing.runner import load_config, run_sweep


def main() -> int:
    run_sweep(
        Path("HRIPCB_UPDATE"),
        Path("runs/member4_validation_sweep"),
        load_config(Path("configs/member4_sweep.yaml")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
