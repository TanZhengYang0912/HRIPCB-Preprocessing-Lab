#!/usr/bin/env python3
"""Aggregate completed member sweep records into one report-friendly dashboard."""

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


MODULES = ("member1", "member2", "member3", "member4", "member5")


def _flatten(records: list[dict]) -> tuple[list[str], list[dict]]:
    parameter_keys = sorted({key for record in records for key in record.get("parameters", {})})
    metric_keys = sorted({key for record in records for key in record.get("metrics", {})})
    fields = ["id", "model_id", "module", "technique", "split", *[f"parameter.{key}" for key in parameter_keys], *[f"metric.{key}" for key in metric_keys]]
    rows = []
    for record in records:
        row = {key: "" for key in fields}
        row.update({
            "id": record.get("id", ""),
            "model_id": record.get("model_id", "baseline"),
            "module": record.get("module", ""),
            "technique": record.get("technique", ""),
            "split": record.get("split", ""),
        })
        row.update({f"parameter.{key}": value for key, value in record.get("parameters", {}).items()})
        row.update({f"metric.{key}": value for key, value in record.get("metrics", {}).items()})
        rows.append(row)
    return fields, rows


def _member5_results_complete(source_dir: Path) -> bool:
    """Return whether a Member 5 output has a durable, complete result set.

    ``results.json`` is a derived public copy.  The progress state is the
    authority for completion, and all three ID sets must agree before a run
    can be included in project-wide reports.
    """

    progress_path = source_dir / "progress.json"
    results_path = source_dir / "results.json"
    if not progress_path.is_file() or not results_path.is_file():
        return False
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        result_records = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(progress, dict) or progress.get("status") != "complete":
        return False
    if not isinstance(result_records, list):
        return False
    if any(
        not isinstance(record, dict) or record.get("module") != "member5"
        for record in result_records
    ):
        return False

    candidate_ids = progress.get("candidate_ids")
    completed_ids = progress.get("completed_ids")
    result_ids = [record.get("id") for record in result_records if isinstance(record, dict)]
    if not isinstance(candidate_ids, list) or not isinstance(completed_ids, list):
        return False
    if not candidate_ids:
        return False
    if len(result_ids) != len(result_records):
        return False
    if any(not isinstance(value, str) or not value for value in (*candidate_ids, *completed_ids, *result_ids)):
        return False
    if any(len(values) != len(set(values)) for values in (candidate_ids, completed_ids, result_ids)):
        return False
    return set(candidate_ids) == set(completed_ids) == set(result_ids)


def _metric_value(record: dict, metric: str = "map50_95") -> float:
    try:
        return float((record.get("metrics") or {}).get(metric, float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def write_project_reports(
    output_root: Path,
    records: list[dict],
    *,
    source_files: tuple[str, ...] | list[str] = (),
) -> Path:
    """Write derived project reports for an already merged record list.

    The caller owns publication of ``results.json``.  Keeping this helper
    separate lets the Member 5 runner atomically merge that file first, then
    refresh CSV, selection, and the static dashboard from the exact same
    records without a second results write.
    """

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = [dict(record) for record in records]
    if not records:
        raise FileNotFoundError("No project records supplied")

    fields, rows = _flatten(records)
    with (output_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    selection_records = [
        record for record in records
        if record.get("split") == "val"
        and record.get("evaluation_type", "ablation") == "ablation"
        and is_combined_record(record)
    ] or records
    ranked_for_selection = sorted(
        selection_records,
        key=lambda record: (_metric_value(record), str(record.get("id", ""))),
        reverse=True,
    )
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
        "source_files": list(source_files),
    }
    (output_root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    write_dashboard_html(
        output_root,
        records,
        title="All Members / Preprocessing Comparison",
        primary_metric="map50_95",
    )
    return output_root


def aggregate_results(runs_root: Path, output_root: Path) -> Path:
    runs_root = Path(runs_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    existing_results = output_root / "results.json"
    existing_records = (
        json.loads(existing_results.read_text(encoding="utf-8"))
        if existing_results.is_file()
        else []
    )
    records: list[dict] = []
    source_files: list[str] = []
    replaced_modules: set[str] = set()
    sources = [
        (
            runs_root / "member2_full_search"
            if module == "member2" and (runs_root / "member2_full_search" / "results.json").is_file()
            else runs_root / "member5_full_search"
            if module == "member5"
            else runs_root / f"{module}_validation_sweep",
            module,
        )
        for module in MODULES
    ]
    official_test_dir = runs_root / "official_test_comparison"
    if (official_test_dir / "results.json").is_file():
        sources.append((official_test_dir, "official_test"))
    for source_dir, source_label in sources:
        source_file = source_dir / "results.json"
        if not source_file.is_file():
            continue
        if source_label == "member5" and not _member5_results_complete(source_dir):
            continue
        if source_label in MODULES:
            replaced_modules.add(source_label)
        source_files.append(str(source_file))
        for record in json.loads(source_file.read_text(encoding="utf-8")):
            record = dict(record)
            if record.get("preview"):
                record["preview"] = os.path.relpath(source_dir / record["preview"], output_root)
            record["source_run"] = str(source_dir)
            records.append(record)
    records.extend(
        record for record in existing_records
        if (
            record.get("module") not in replaced_modules
            or (
                record.get("module") == "member5"
                and not (
                    record.get("split") == "val"
                    and record.get("evaluation_type", "ablation") == "ablation"
                )
            )
        )
    )
    if not records:
        raise FileNotFoundError("No member validation results found under runs/")

    (output_root / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return write_project_reports(output_root, records, source_files=source_files)


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
