#!/usr/bin/env python3
"""Run the extensible Member 1 preprocessing parameter sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_member1.sweep_runner import load_sweep_config, run_member1_sweep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("HRIPCB_UPDATE"))
    parser.add_argument("--output", type=Path, default=Path("runs/member1_parameter_sweep"))
    parser.add_argument("--config", type=Path, default=Path("configs/member1_sweep.yaml"))
    args = parser.parse_args()
    output = run_member1_sweep(args.dataset, args.output, load_sweep_config(args.config))
    print(f"parameter sweep output: {output}")
    print(f"dashboard: {output / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
