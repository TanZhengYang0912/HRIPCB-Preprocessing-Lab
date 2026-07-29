#!/usr/bin/env python3
"""Run one generic Member 1-4 preprocessing sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_preprocessing.runner import load_config, run_sweep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("HRIPCB_UPDATE"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output = run_sweep(args.dataset, args.output, load_config(args.config))
    print(f"parameter sweep output: {output}")
    print(f"dashboard: {output / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
