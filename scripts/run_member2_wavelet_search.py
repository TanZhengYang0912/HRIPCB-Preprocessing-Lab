#!/usr/bin/env python3
"""Run Member 2's coarse-to-fine Wavelet parameter search."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_preprocessing.runner import load_config, run_sweep


def top_wavelets(records: list[dict], count: int) -> list[dict]:
    wavelets = [record for record in records if record.get("technique") == "wavelet"]
    ranked = sorted(
        wavelets,
        key=lambda record: float(record.get("metrics", {}).get("map50_95", float("-inf"))),
        reverse=True,
    )
    if len(ranked) < count:
        raise ValueError(f"Expected at least {count} Wavelet results, found {len(ranked)}")
    return ranked[:count]


def build_refinement_presets(winners: list[dict], refinement: dict) -> list[dict]:
    presets = []
    combinations = product(
        winners,
        refinement["methods"],
        refinement["modes"],
        refinement["levels"],
    )
    for winner, method, mode, level in combinations:
        wavelet = str(winner["parameters"]["wavelet_name"])
        method_id = str(method).lower().removesuffix("shrink")
        level_id = "auto" if level is None else str(level)
        preset = {
            "id": f"{wavelet}_{method_id}_{mode}_l{level_id}",
            "wavelet": wavelet,
            "method": str(method),
            "mode": str(mode),
        }
        if level is not None:
            preset["wavelet_levels"] = int(level)
        presets.append(preset)
    return presets


def _run(dataset: Path, output: Path, base_config: dict, presets: list[dict]) -> list[dict]:
    config = {
        **base_config,
        "wavelet_presets": presets,
        "homomorphic_presets": [],
    }
    run_sweep(dataset, output, config)
    return json.loads((output / "results.json").read_text(encoding="utf-8"))


def run_search(dataset: Path, output: Path, config: dict) -> Path:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    coarse_records = _run(
        dataset,
        output / "coarse",
        config,
        list(config["coarse_wavelets"]),
    )
    coarse_winners = top_wavelets(coarse_records, int(config.get("top_wavelets", 2)))
    refinement_presets = build_refinement_presets(coarse_winners, config["refinement"])
    refinement_records = _run(
        dataset,
        output / "refinement",
        config,
        refinement_presets,
    )
    best = top_wavelets(refinement_records, 1)[0]
    original = next(record for record in coarse_records if record.get("technique") == "original")
    summary = {
        "primary_metric": "map50_95",
        "coarse_candidate_count": len(coarse_records),
        "refinement_candidate_count": len(refinement_records),
        "coarse_winners": coarse_winners,
        "best_wavelet": best,
        "original_map50_95": float(original["metrics"]["map50_95"]),
        "improvement_vs_original": (
            float(best["metrics"]["map50_95"])
            - float(original["metrics"]["map50_95"])
        ),
    }
    summary_path = output / "stage1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("HRIPCB_UPDATE"))
    parser.add_argument("--output", type=Path, default=Path("runs/member2_wavelet_search"))
    parser.add_argument("--config", type=Path, default=Path("configs/member2_wavelet_search.yaml"))
    args = parser.parse_args()
    summary_path = run_search(args.dataset, args.output, load_config(args.config))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary["best_wavelet"]
    print(f"stage 1 summary: {summary_path}")
    print(f"best Wavelet: {best['id']} / mAP50-95={best['metrics']['map50_95']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
