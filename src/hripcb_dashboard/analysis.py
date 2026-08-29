"""Project-specific analysis data for the HRIPCB dashboard."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .filtering import best_by_module, collapse_shared_baseline, is_combined_record


TECHNIQUE_LABELS = {
    "original": "Original",
    "gaussian": "Gaussian",
    "bbhe": "BBHE",
    "gaussian_bbhe": "Gaussian + BBHE",
    "wavelet": "Wavelet",
    "homomorphic": "Homomorphic",
    "wavelet_homomorphic": "Wavelet + Homomorphic",
    "bilateral": "Bilateral",
    "agcwd": "AGCWD",
    "bilateral_agcwd": "Bilateral + AGCWD",
    "nlm": "Non-local Means",
    "msr": "Multi-Scale Retinex",
    "nlm_msr": "NLM + MSR",
}

MEMBER_TECHNIQUES = {
    "member1": ("gaussian", "bbhe"),
    "member2": ("wavelet", "homomorphic"),
    "member3": ("bilateral", "agcwd"),
    "member4": ("nlm", "msr"),
}

ANALYSIS_METRICS = ("precision", "recall", "map50", "map50_95", "f1")
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "map50": "mAP50",
    "map50_95": "mAP50-95",
    "f1": "F1",
}


def technique_label(technique: object) -> str:
    """Return a concise human-readable technique name."""

    value = str(technique or "unknown").lower()
    return TECHNIQUE_LABELS.get(value, value.replace("_", " + ").title())


def metric_label(metric: object) -> str:
    """Return a consistent label for a detection metric."""

    value = str(metric or "")
    return METRIC_LABELS.get(value, value.replace("_", " ").title())


def _metric(record: Mapping[str, object], key: str) -> float:
    try:
        return float((record.get("metrics") or {}).get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_val_ablation(record: Mapping[str, object]) -> bool:
    return (
        str(record.get("split", "")) == "val"
        and str(record.get("evaluation_type", "ablation")) == "ablation"
    )


def _chart_row(record: Mapping[str, object], *, label: str) -> dict:
    return {
        "label": label,
        "id": str(record.get("id", "")),
        "module": str(record.get("module", "")),
        "technique": technique_label(record.get("technique")),
        "parameters": dict(record.get("parameters") or {}),
        **{key: _metric(record, key) for key in ANALYSIS_METRICS},
    }


def _original_control(records: list[dict]) -> dict | None:
    val_originals = [
        record for record in collapse_shared_baseline(records)
        if str(record.get("split", "")) == "val"
        and str(record.get("technique", "")).lower() == "original"
    ]
    return val_originals[0] if val_originals else None


def _best_single(records: list[dict], module: str, technique: str) -> dict | None:
    candidates = [
        record for record in records
        if _is_val_ablation(record)
        and str(record.get("module", "")) == module
        and str(record.get("technique", "")).lower() == technique
        and not is_combined_record(record)
    ]
    return max(candidates, key=lambda record: (_metric(record, "map50_95"), str(record.get("id", ""))), default=None)


def build_analysis_payload(records: Iterable[Mapping[str, object]]) -> dict:
    """Build all report-ready datasets used by Analysis & reports.

    The primary comparison is deliberately five rows: one shared original
    control and one best combined result for each of the four member modules.
    Single noise/contrast runs remain in the reference data and stage table.
    """

    source = [dict(record) for record in records]
    combined_winner_records = best_by_module(source, split="val", evaluation_type="ablation")
    original = _original_control(source)

    original_vs_combined: list[dict] = []
    if original is not None:
        original_vs_combined.append(_chart_row(original, label="Original"))
    original_vs_combined.extend(
        _chart_row(
            record,
            label=f"{record.get('module', 'unknown')} / {technique_label(record.get('technique'))}",
        )
        for record in combined_winner_records
    )
    original_vs_combined.sort(
        key=lambda row: (row["map50_95"], row["id"]),
        reverse=True,
    )

    metric_comparison = [
        {
            "label": f"{record.get('module', 'unknown')} / {technique_label(record.get('technique'))}",
            "id": str(record.get("id", "")),
            "module": str(record.get("module", "")),
            "technique": technique_label(record.get("technique")),
            **{key: _metric(record, key) for key in ANALYSIS_METRICS},
        }
        for record in combined_winner_records
    ]
    metric_comparison.sort(
        key=lambda row: (row["map50_95"], row["id"]),
        reverse=True,
    )

    stage_comparison = []
    for module, (noise_technique, contrast_technique) in MEMBER_TECHNIQUES.items():
        row = {"member": module}
        if original is not None:
            row["Original"] = _metric(original, "map50_95")
        for stage, technique in (("Noise-only", noise_technique), ("Contrast-only", contrast_technique)):
            single = _best_single(source, module, technique)
            row[stage] = _metric(single, "map50_95") if single else None
        combined = next((record for record in combined_winner_records if record.get("module") == module), None)
        row["Combined"] = _metric(combined, "map50_95") if combined else None
        stage_comparison.append(row)

    sensitivity = []
    for record in source:
        if not (_is_val_ablation(record) and is_combined_record(record)):
            continue
        sensitivity.append({
            "id": str(record.get("id", "")),
            "module": str(record.get("module", "")),
            "technique": technique_label(record.get("technique")),
            "mAP50-95": _metric(record, "map50_95"),
            "parameters": dict(record.get("parameters") or {}),
        })
    sensitivity.sort(key=lambda row: (row["module"], -row["mAP50-95"], row["id"]))

    baseline_test = next(
        (record for record in source if record.get("id") == "baseline_original_test"),
        None,
    )
    retrained_test = next(
        (record for record in source if record.get("evaluation_type") == "retrained_candidate"),
        None,
    )
    retrained_vs_baseline = []
    if baseline_test is not None and retrained_test is not None:
        for metric in ANALYSIS_METRICS:
            retrained_vs_baseline.append({
                "Metric": metric_label(metric),
                "Baseline": _metric(baseline_test, metric),
                "Retrained candidate": _metric(retrained_test, metric),
                "Difference": _metric(retrained_test, metric) - _metric(baseline_test, metric),
            })

    return {
        "original_vs_combined": original_vs_combined,
        "combined_winners": [_chart_row(record, label=f"{record.get('module', 'unknown')} / {technique_label(record.get('technique'))}") for record in combined_winner_records],
        "metric_comparison": metric_comparison,
        "stage_comparison": stage_comparison,
        "parameter_sensitivity": sensitivity,
        "retrained_vs_baseline": retrained_vs_baseline,
    }
