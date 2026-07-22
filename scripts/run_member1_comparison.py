#!/usr/bin/env python3
"""Run Member 1's Gaussian Filtering and BBHE image comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_member1.runner import load_member1_config, run_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("HRIPCB_UPDATE"))
    parser.add_argument("--output", type=Path, default=Path("runs/member1"))
    parser.add_argument("--config", type=Path, default=Path("configs/member1.yaml"))
    parser.add_argument("--sample", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_member1_config(args.config)
    output = run_comparison(args.dataset, args.output, args.sample, config)
    print(f"member1 comparison output: {output}")
    print(f"source count: {len(list((args.dataset / config['split'] / 'images').glob('*')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
