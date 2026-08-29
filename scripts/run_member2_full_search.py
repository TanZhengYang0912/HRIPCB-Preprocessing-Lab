#!/usr/bin/env python3
"""Run stage 2 of Member 2's Wavelet + Homomorphic search."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_dashboard.dashboard import write_dashboard_html
from hripcb_preprocessing.runner import _write_flat_csv, load_config, run_sweep


def _label(value: float) -> str:
    return str(value).replace(".", "p")


def build_homomorphic_presets(grid: dict) -> list[dict]:
    presets = []
    for gain_pair, cutoff, sharpness in product(
        grid["gain_pairs"], grid["cutoffs"], grid["sharpness"]
    ):
        gamma_low, gamma_high = map(float, gain_pair)
        cutoff = float(cutoff)
        sharpness = float(sharpness)
        presets.append({
            "id": (
                f"gl{_label(gamma_low)}_gh{_label(gamma_high)}_"
                f"c{_label(cutoff)}_s{_label(sharpness)}"
            ),
            "gamma_low": gamma_low,
            "gamma_high": gamma_high,
            "cutoff": cutoff,
            "sharpness": sharpness,
        })
    return presets


def best_by_technique(records: list[dict], technique: str) -> dict:
    matches = [record for record in records if record.get("technique") == technique]
    if not matches:
        raise ValueError(f"No results found for technique: {technique}")
    return max(matches, key=lambda record: float(record["metrics"]["map50_95"]))


def partition_presets(presets: list[dict], count: int) -> list[list[dict]]:
    size = max(1, (len(presets) + count - 1) // count)
    return [presets[index:index + size] for index in range(0, len(presets), size)]


def merge_shard_records(shards: list[tuple[str, list[dict]]]) -> list[dict]:
    merged = {}
    for shard_name, records in shards:
        for record in records:
            merged.setdefault(
                record["id"],
                {
                    **record,
                    "preview": f"shards/{shard_name}/{record['preview']}",
                },
            )
    return list(merged.values())


def _run_shard(task: tuple[Path, Path, dict]) -> tuple[str, list[dict]]:
    dataset, output, config = task
    run_sweep(dataset, output, config)
    records = json.loads((output / "results.json").read_text(encoding="utf-8"))
    return output.name, records


def _rank(records: list[dict], technique: str, count: int = 5) -> list[dict]:
    matches = [record for record in records if record.get("technique") == technique]
    return sorted(
        matches,
        key=lambda record: float(record["metrics"]["map50_95"]),
        reverse=True,
    )[:count]


def run_search(
    dataset: Path,
    output: Path,
    config: dict,
    stage1_summary_path: Path,
) -> Path:
    stage1 = json.loads(stage1_summary_path.read_text(encoding="utf-8"))
    winner = stage1["best_wavelet"]
    parameters = winner["parameters"]
    wavelet_preset = {
        "id": "stage1_winner",
        "wavelet": parameters["wavelet_name"],
        "method": parameters["wavelet_method"],
        "mode": parameters["wavelet_mode"],
    }
    if parameters.get("wavelet_levels") is not None:
        wavelet_preset["wavelet_levels"] = int(parameters["wavelet_levels"])

    output = output.resolve()
    homomorphic_presets = build_homomorphic_presets(config["homomorphic_grid"])
    workers = int(config.get("parallel_workers", 1))
    if workers > 1:
        tasks = []
        for index, presets in enumerate(partition_presets(homomorphic_presets, workers), start=1):
            shard_output = output / "shards" / f"part_{index:02d}"
            tasks.append((
                dataset,
                shard_output,
                {
                    **config,
                    "wavelet_presets": [wavelet_preset],
                    "homomorphic_presets": presets,
                },
            ))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            shard_records = list(executor.map(_run_shard, tasks))
        records = merge_shard_records(shard_records)
        (output / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        _write_flat_csv(output / "results.csv", records)
        write_dashboard_html(
            output,
            records,
            title="Member2 / Wavelet + Homomorphic Full Search",
            primary_metric="map50_95",
        )
    else:
        run_sweep(
            dataset,
            output,
            {
                **config,
                "wavelet_presets": [wavelet_preset],
                "homomorphic_presets": homomorphic_presets,
            },
        )
        records = json.loads((output / "results.json").read_text(encoding="utf-8"))
    original = best_by_technique(records, "original")
    best_homomorphic = best_by_technique(records, "homomorphic")
    best_combined = best_by_technique(records, "wavelet_homomorphic")
    summary = {
        "primary_metric": "map50_95",
        "candidate_count": len(records),
        "stage1_best_wavelet": winner,
        "best_homomorphic": best_homomorphic,
        "best_combined": best_combined,
        "top_homomorphic": _rank(records, "homomorphic"),
        "top_combined": _rank(records, "wavelet_homomorphic"),
        "original_map50_95": float(original["metrics"]["map50_95"]),
        "combined_improvement_vs_original": (
            float(best_combined["metrics"]["map50_95"])
            - float(original["metrics"]["map50_95"])
        ),
    }
    summary_path = output / "stage2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("HRIPCB_UPDATE"))
    parser.add_argument("--output", type=Path, default=Path("runs/member2_full_search"))
    parser.add_argument("--config", type=Path, default=Path("configs/member2_full_search.yaml"))
    parser.add_argument(
        "--stage1-summary",
        type=Path,
        default=Path("runs/member2_wavelet_search/stage1_summary.json"),
    )
    args = parser.parse_args()
    summary_path = run_search(
        args.dataset,
        args.output,
        load_config(args.config),
        args.stage1_summary,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary["best_combined"]
    print(f"stage 2 summary: {summary_path}")
    print(f"best combination: {best['id']} / mAP50-95={best['metrics']['map50_95']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
