#!/usr/bin/env python3
"""Aggregate Member 1-4 sweep records into one report-friendly dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_dashboard.dashboard import write_dashboard_html
from hripcb_dashboard.filtering import is_combined_record


MODULES = ("member1", "member2", "member3", "member4")


def _flatten(records: list[dict]) -> tuple[list[str], list[dict]]:
    parameter_keys = sorted({key for record in records for key in record.get("parameters", {})})
    metric_keys = sorted({key for record in records for key in record.get("metrics", {})})
    fields = ["id", "model_id", "module", "technique", "split", *[f"parameter.{key}" for key in parameter_keys], *[f"metric.{key}" for key in metric_keys]]
    rows = []
    for record in records:
        row = {key: "" for key in fields}
        row.update({"id": record["id"], "model_id": record.get("model_id", "baseline"), "module": record["module"], "technique": record["technique"], "split": record.get("split", "")})
        row.update({f"parameter.{key}": value for key, value in record.get("parameters", {}).items()})
        row.update({f"metric.{key}": value for key, value in record.get("metrics", {}).items()})
        rows.append(row)
    return fields, rows


def aggregate_results(runs_root: Path, output_root: Path) -> Path:
    runs_root = Path(runs_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    source_files: list[str] = []
    sources = [(runs_root / f"{module}_validation_sweep", module) for module in MODULES]
    official_test_dir = runs_root / "official_test_comparison"
    if (official_test_dir / "results.json").is_file():
        sources.append((official_test_dir, "official_test"))
    for source_dir, _source_label in sources:
        source_file = source_dir / "results.json"
        if not source_file.is_file():
            continue
        source_files.append(str(source_file))
        for record in json.loads(source_file.read_text(encoding="utf-8")):
            record = dict(record)
            if record.get("preview"):
                record["preview"] = os.path.relpath(source_dir / record["preview"], output_root)
            record["source_run"] = str(source_dir)
            records.append(record)
    if not records:
        raise FileNotFoundError("No member validation results found under runs/")

    (output_root / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    fields, rows = _flatten(records)
    with (output_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(records, key=lambda record: float(record.get("metrics", {}).get("map50_95", float("-inf"))), reverse=True)
    selection_records = [
        record for record in records
        if record.get("split") == "val"
        and record.get("evaluation_type", "ablation") == "ablation"
        and is_combined_record(record)
    ] or records
    ranked_for_selection = sorted(selection_records, key=lambda record: float(record.get("metrics", {}).get("map50_95", float("-inf"))), reverse=True)
    by_module = {}
    for module in MODULES:
        module_rows = [record for record in ranked_for_selection if record.get("module") == module]
        if module_rows:
            by_module[module] = module_rows[0]
    selection = {
        "primary_metric": "map50_95",
        "selection_split": "val",
        "overall_best": ranked_for_selection[0],
        "best_by_module": by_module,
        "record_count": len(records),
        "source_files": source_files,
    }
    (output_root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    write_dashboard_html(
        output_root,
        records,
        title="All Members / Preprocessing Comparison",
        primary_metric="map50_95",
    )
    return output_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("runs/project_validation_comparison"))
    args = parser.parse_args()
    output = aggregate_results(args.runs_root, args.output)
    print(f"project dashboard: {output / 'dashboard.html'}")
    print(f"project results: {output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
