#!/usr/bin/env python3
"""Validate the local HRIPCB dataset without changing it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_baseline.dataset import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    report = validate_dataset(args.root)
    print(f"total_images: {report.total_images}")
    print(f"total_labels: {report.total_labels}")
    print(f"split_counts: {report.split_counts}")
    print(f"class_counts: {report.class_counts}")
    if report.errors:
        print("errors:")
        for error in report.errors:
            print(f"- {error}")
        return 1
    print("status: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
