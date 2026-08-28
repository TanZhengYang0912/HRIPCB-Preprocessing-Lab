"""Shared, dependency-free filtering rules for the HRIPCB dashboards."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping

ALL = "all"
FILTER_FIELDS = ("model", "split", "module", "technique")
COMBINED_TECHNIQUES = frozenset({
    "gaussian_bbhe",
    "wavelet_homomorphic",
    "bilateral_agcwd",
    "nlm_msr",
})
MEMBER_MODULES = ("member1", "member2", "member3", "member4")


def is_shared_original_record(record: Mapping[str, object]) -> bool:
    """Return whether a record is one of the duplicated member original controls."""

    return (
        str(record.get("model_id", "baseline")) == "baseline"
        and str(record.get("split", "")) == "val"
        and str(record.get("technique", "")).lower() == "original"
        and str(record.get("module", "")) in MEMBER_MODULES
    )


def collapse_shared_baseline(records: Iterable[Mapping[str, object]]) -> list[dict]:
    """Collapse duplicated member original rows into one shared baseline control."""

    source = [dict(record) for record in records]
    originals = [record for record in source if is_shared_original_record(record)]
    if len(originals) < 2:
        return source

    control = dict(originals[0])
    control.update({
        "id": "original_shared_control",
        "module": "baseline",
        "technique": "original",
        "evaluation_type": "baseline_control",
        "shared_control_modules": sorted({str(record.get("module")) for record in originals}),
        "display_label": "Original / Shared baseline control",
    })
    collapsed: list[dict] = []
    inserted = False
    for record in source:
        if is_shared_original_record(record):
            if not inserted:
                collapsed.append(control)
                inserted = True
            continue
        collapsed.append(record)
    return collapsed


def comparison_records(
    records: Iterable[Mapping[str, object]],
    *,
    model: str = ALL,
    split: str = ALL,
    module: str = ALL,
    technique: str = ALL,
    search: str = "",
    run_type: str = ALL,
) -> list[dict]:
    """Return display rows, adding the original baseline to combined comparisons."""

    if run_type == "combined":
        selected = filter_records(
            records,
            model=model,
            split=split,
            module=module,
            technique=technique,
            search=search,
            run_type="combined",
        )
        selected.extend(
            filter_records(
                records,
                model=model,
                split=split,
                module=module,
                technique="original",
                search=search,
                run_type=ALL,
            )
        )
    else:
        selected = filter_records(
            records,
            model=model,
            split=split,
            module=module,
            technique=technique,
            search=search,
            run_type=run_type,
        )
    return collapse_shared_baseline(selected) if module == ALL else selected


def is_combined_record(record: Mapping[str, object]) -> bool:
    """Return whether a record applies both techniques for its member module."""

    if record.get("is_combined") is not None:
        return bool(record.get("is_combined"))
    if str(record.get("evaluation_stage", "")).lower() == "combined":
        return True
    return str(record.get("technique", "")).lower() in COMBINED_TECHNIQUES


def inference_widget_keys(prefix: str) -> tuple[str, str, str]:
    """Return isolated model/module/technique keys for one inference tab."""

    return tuple(f"{prefix}_{field}" for field in ("model", "module", "technique"))


def _value(record: Mapping[str, object], field: str) -> str:
    if field == "run_type":
        return "combined" if is_combined_record(record) else "reference"
    key = "model_id" if field == "model" else field
    value = record.get(key)
    return str(value) if value not in (None, "") else "unknown"


def _matches(record: Mapping[str, object], selection: Mapping[str, str]) -> bool:
    return all(
        selection.get(field, ALL) == ALL
        or _value(record, field) == selection.get(field)
        for field in FILTER_FIELDS
    )


def filter_records(
    records: Iterable[Mapping[str, object]],
    *,
    model: str = ALL,
    split: str = ALL,
    module: str = ALL,
    technique: str = ALL,
    search: str = "",
    run_type: str = ALL,
) -> list[dict]:
    """Return records matching all active filters, preserving input order."""

    selection = {"model": model, "split": split, "module": module, "technique": technique}
    query = search.strip().lower()
    filtered: list[dict] = []
    for record in records:
        if not _matches(record, selection):
            continue
        if run_type != ALL and _value(record, "run_type") != run_type:
            continue
        if query and query not in json.dumps(record, sort_keys=True, default=str).lower():
            continue
        filtered.append(dict(record))
    return filtered


def option_values(
    records: Iterable[Mapping[str, object]],
    *,
    model: str = ALL,
    split: str = ALL,
    module: str = ALL,
) -> dict[str, list[str]]:
    """Return valid values for each filter after applying upstream choices."""

    source = list(records)
    model_records = filter_records(source, model=model)
    split_records = filter_records(model_records, model=model, split=split)
    module_records = filter_records(split_records, model=model, split=split, module=module)
    return {
        "model": sorted({_value(record, "model") for record in source}),
        "split": sorted({_value(record, "split") for record in model_records}),
        "module": sorted({_value(record, "module") for record in split_records}),
        "technique": sorted({_value(record, "technique") for record in module_records}),
    }


def normalize_selection(
    records: Iterable[Mapping[str, object]], selection: Mapping[str, str]
) -> dict[str, str]:
    """Reset invalid selections while walking the filter cascade in order."""

    source = list(records)
    normalized = {field: selection.get(field, ALL) for field in FILTER_FIELDS}

    for field in FILTER_FIELDS:
        upstream = {key: normalized[key] for key in FILTER_FIELDS if FILTER_FIELDS.index(key) < FILTER_FIELDS.index(field)}
        options = option_values(
            source,
            model=upstream.get("model", ALL),
            split=upstream.get("split", ALL),
            module=upstream.get("module", ALL),
        )[field]
        if normalized[field] != ALL and normalized[field] not in options:
            normalized[field] = ALL

    return normalized


def reset_selection_state(
    state: MutableMapping[str, object],
    *,
    prefix: str,
    extra_fields: Iterable[str] = (),
    defaults: Mapping[str, object] | None = None,
) -> None:
    """Reset only filter-widget keys belonging to one UI selection group."""

    defaults = defaults or {}
    for field in (*FILTER_FIELDS, *extra_fields):
        key = f"{prefix}{field}"
        if key in state:
            state[key] = defaults.get(field, ALL)


def _metric_value(record: Mapping[str, object], metric: str) -> float:
    try:
        return float((record.get("metrics") or {}).get(metric, float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _recommendation_candidates(
    records: Iterable[Mapping[str, object]],
    *,
    split: str,
    evaluation_type: str,
) -> list[dict]:
    return [
        dict(record)
        for record in records
        if _value(record, "split") == split
        and str(record.get("evaluation_type", "ablation")) == evaluation_type
        and is_combined_record(record)
        and _metric_value(record, "map50_95") != float("-inf")
    ]


def best_experiment(
    records: Iterable[Mapping[str, object]],
    *,
    split: str = "val",
    evaluation_type: str = "ablation",
    metric: str = "map50_95",
) -> dict | None:
    """Return the highest-scoring combined recommendation for the frozen protocol."""

    candidates = [
        record
        for record in records
        if _value(record, "split") == split
        and str(record.get("evaluation_type", "ablation")) == evaluation_type
        and is_combined_record(record)
        and _metric_value(record, metric) != float("-inf")
    ]
    return max(candidates, key=lambda record: (_metric_value(record, metric), record.get("id", "")), default=None)


def best_by_module(
    records: Iterable[Mapping[str, object]],
    *,
    split: str = "val",
    evaluation_type: str = "ablation",
    metric: str = "map50_95",
) -> list[dict]:
    """Return one highest-scoring recommendation per module in stable module order."""

    candidates = _recommendation_candidates(
        records, split=split, evaluation_type=evaluation_type
    )
    candidates_by_module: dict[str, list[dict]] = {}
    for record in candidates:
        candidates_by_module.setdefault(_value(record, "module"), []).append(record)
    return [
        max(rows, key=lambda record: (_metric_value(record, metric), record.get("id", "")))
        for module, rows in sorted(candidates_by_module.items())
    ]
