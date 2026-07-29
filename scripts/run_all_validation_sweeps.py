#!/usr/bin/env python3
"""Run the same validation protocol for Members 1-4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_preprocessing.runner import load_config, run_sweep


MODULE_CONFIGS = (
    ("member1", Path("configs/member1_validation_sweep.yaml")),
    ("member2", Path("configs/member2_sweep.yaml")),
    ("member3", Path("configs/member3_sweep.yaml")),
    ("member4", Path("configs/member4_sweep.yaml")),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("HRIPCB_UPDATE"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    for index, (module, config_path) in enumerate(MODULE_CONFIGS, start=1):
        print(f"[{index}/{len(MODULE_CONFIGS)}] starting {module}", flush=True)
        run_sweep(args.dataset, args.output_root / f"{module}_validation_sweep", load_config(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
