#!/usr/bin/env python3
"""Manually launch the resumable Member 5 validation search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hripcb_preprocessing.member5_search import run_search
from hripcb_preprocessing.runner import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("HRIPCB_UPDATE"))
    parser.add_argument("--output", type=Path, default=Path("runs/member5_full_search"))
    parser.add_argument("--config", type=Path, default=Path("configs/member5_full_search.yaml"))
    parser.add_argument("--batch-size", type=int, help="Candidates evaluated per batch (default: 2)")
    parser.add_argument("--keep-variants", action="store_true", help="Keep full processed datasets and evaluation staging")
    parser.add_argument("--project-results", type=Path, default=Path("runs/project_validation_comparison/results.json"))
    args = parser.parse_args()
    try:
        summary_path = run_search(
            args.dataset, args.output, load_config(args.config), batch_size=args.batch_size,
            keep_variants=args.keep_variants, project_results=args.project_results,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Member 5 search: {error}\n")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary["best_combined"]
    print(f"summary: {summary_path}")
    print(f"best combination: {best['id']} / mAP50-95={best['metrics']['map50_95']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
