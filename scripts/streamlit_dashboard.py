#!/usr/bin/env python3
"""Interactive Streamlit view over the same generic project results JSON."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import altair as alt
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_RESULTS_PATH = PROJECT_ROOT / "runs/project_validation_comparison/results.json"

from hripcb_member1.evaluation import select_device
from hripcb_dashboard.batch import extract_image_entries
from hripcb_dashboard.analysis import (
    MEMBER_TECHNIQUES,
    build_analysis_payload,
    module_label,
    ranking_chart_rows,
    technique_label,
)
from hripcb_dashboard.reporting import build_report_pdf, dumps_json, record_metric_summary
from hripcb_dashboard.video import process_video
from hripcb_preprocessing.candidates import apply_candidate
from hripcb_dashboard.filtering import (
    FILTER_FIELDS,
    best_by_module,
    best_experiment,
    comparison_records,
    filter_records,
    inference_widget_keys,
    is_combined_record,
    normalize_selection,
    option_values,
    reset_selection_state,
)


METRIC_LABELS = {
    "map50_95": "mAP50-95",
    "map50": "mAP50",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mean_psnr": "Mean PSNR",
    "mean_ssim": "Mean SSIM",
    "milliseconds": "Time (ms)",
}

INFERENCE_IMGSZ = 1024
INFERENCE_CONF = 0.25
INFERENCE_IOU = 0.70
INFERENCE_MAX_SIDE = 1280
IMAGE_UPLOAD_VERSION_KEY = "image_inference_uploader_version"
MEMBER5_PARAMETER_KEYS = (
    "tv_weight",
    "morphology_kernel_size",
    "top_hat_amount",
    "black_hat_amount",
)


def _load_records(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_results_path(path: Path) -> Path:
    """Resolve relative result paths from the repository, not the launch cwd."""

    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _label(key: str) -> str:
    return METRIC_LABELS.get(key, key.replace("_", " ").title())


def _checkpoint_for_model(model_id: str) -> Path:
    return PROJECT_ROOT / "runs/baseline/weights/best.pt"


def _candidate_from_record(record: dict) -> dict:
    return {
        "module": record.get("module", "member1"),
        "technique": record.get("technique", "original"),
        "parameters": record.get("parameters", {}),
    }


def _protocol_payload() -> dict:
    return {
        "imgsz": INFERENCE_IMGSZ,
        "conf": INFERENCE_CONF,
        "iou": INFERENCE_IOU,
        "workers": 0,
        "seed": 42,
        "primary_metric": "map50_95",
    }


def _metric_value(record: dict, key: str) -> float:
    try:
        return float((record.get("metrics") or {}).get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _resolve_preview_path(results_path: Path, preview: str | None) -> Path | None:
    if not preview:
        return None
    candidate = (results_path.parent / preview).resolve()
    return candidate if candidate.is_file() else None


def _render_report_tools(st, records: list[dict]) -> None:
    st.subheader("Report & export")
    st.caption("Download the experiment evidence for your report or presentation.")
    try:
        pdf_payload = build_report_pdf(records, _protocol_payload())
    except ImportError as error:
        st.error(f"PDF dependency unavailable: {error}")
        pdf_payload = None
    export_col, data_col = st.columns(2)
    with export_col:
        if pdf_payload:
            st.download_button(
                "Download complete PDF report",
                data=pdf_payload,
                file_name="hripcb_preprocessing_report_v2.pdf",
                mime="application/pdf",
                key="download_pdf_report",
            )
    with data_col:
        st.download_button(
            "Download reproducibility JSON",
            data=dumps_json({"protocol": _protocol_payload(), "records": records}),
            file_name="hripcb_reproducibility.json",
            mime="application/json",
            key="download_reproducibility_json",
        )
    st.download_button(
        "Download complete results CSV",
        data=_records_to_csv(records),
        file_name="hripcb_experiment_results.csv",
        mime="text/csv",
        key="download_results_csv",
    )


def _records_to_csv(records: list[dict]) -> str:
    import csv
    import io

    metric_keys = sorted({key for record in records for key in (record.get("metrics") or {})})
    parameter_keys = sorted({key for record in records for key in (record.get("parameters") or {})})
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "model", "module", "technique", "split", *parameter_keys, *metric_keys])
    for record in records:
        writer.writerow([
            record.get("id", ""),
            record.get("model_id", "baseline"),
            record.get("module", ""),
            record.get("technique", ""),
            record.get("split", ""),
            *[(record.get("parameters") or {}).get(key, "") for key in parameter_keys],
            *[(record.get("metrics") or {}).get(key, "") for key in metric_keys],
        ])
    return output.getvalue()


def _render_analysis(st, records: list[dict]) -> None:
    import pandas as pd

    st.header("Analysis & findings")
    st.caption("A report-ready view of the original control, single-technique references, and four combined candidates.")
    summary = record_metric_summary(records)
    payload = build_analysis_payload(records)
    cards = st.columns(4)
    cards[0].metric("Displayed runs", summary["display_count"])
    cards[1].metric("Combined runs", summary["combined_count"])
    cards[2].metric("Reference runs", summary["reference_count"])
    best = summary["best"]
    cards[3].metric("Best combined mAP50-95", f"{_metric_value(best, 'map50_95'):.4f}" if best else "—")
    st.caption(
        f"Displayed count removes duplicate member original controls. Raw records: {summary['count']}. "
        f"Coverage: {summary['module_count']} member modules + {summary['extra_study_count']} extra study + {summary['baseline_control_count']} baseline controls."
    )
    if best:
        st.success(
            f"Best combined run: {best.get('id', '—')} · "
            f"{module_label(best.get('module'))} / {technique_label(best.get('technique'))} · "
            f"mAP50-95={_metric_value(best, 'map50_95'):.4f}"
        )

    st.subheader("Primary comparison: Original vs four combined techniques")
    st.caption("One shared Original control plus the highest validation mAP50-95 combined result from each member.")
    primary_rows = []
    for rank, row in enumerate(payload["original_vs_combined"], start=1):
        primary_rows.append({
            "Rank": rank,
            "Comparison": row["label"],
            "Experiment": row["id"],
            "mAP50-95": round(row["map50_95"], 4),
            "F1": round(row["f1"], 4),
            "Precision": round(row["precision"], 4),
            "Recall": round(row["recall"], 4),
        })
    st.dataframe(primary_rows, width="stretch", hide_index=True)
    primary_chart = pd.DataFrame(payload["original_vs_combined"])
    if not primary_chart.empty:
        st.bar_chart(primary_chart.set_index("label")[["map50_95"]].rename(columns={"map50_95": "mAP50-95"}), height=300)

    left, right = st.columns(2)
    with left:
        st.subheader("Four combined techniques: metric comparison")
        st.caption("All four winners use the same validation split and frozen detector protocol.")
        metric_chart = pd.DataFrame(payload["metric_comparison"])
        if not metric_chart.empty:
            metric_frame = metric_chart.set_index("label")[
                ["precision", "recall", "map50", "map50_95", "f1"]
            ].rename(columns={
                "precision": "Precision",
                "recall": "Recall",
                "map50": "mAP50",
                "map50_95": "mAP50-95",
                "f1": "F1",
            })
            st.bar_chart(
                metric_frame,
                height=320,
            )
    with right:
        st.subheader("Processing stage comparison")
        st.caption("Best available result for each member's Original, noise-only, contrast-only, and combined stages.")
        stage_chart = pd.DataFrame(payload["stage_comparison"])
        if not stage_chart.empty:
            st.bar_chart(stage_chart.drop(columns=["member"]).set_index("member_label"), height=320)

    st.subheader("Combined parameter sensitivity")
    st.caption("This exposes every combined parameter run so the best setting is auditable, not hidden behind one score.")
    sensitivity = pd.DataFrame(payload["parameter_sensitivity"])
    if not sensitivity.empty:
        modules = sorted(sensitivity["module"].unique())
        selected_module = st.selectbox(
            "Member", modules, key="analysis_sensitivity_module", format_func=module_label
        )
        module_sensitivity = sensitivity[sensitivity["module"] == selected_module].copy()
        module_sensitivity["Parameters"] = module_sensitivity["parameters"].map(lambda value: json.dumps(value, sort_keys=True))
        st.bar_chart(module_sensitivity.set_index("id")[["mAP50-95"]], height=280)
        st.dataframe(module_sensitivity[["id", "technique", "Parameters", "mAP50-95"]], width="stretch", hide_index=True)

    reference_rows = [record for record in summary["all_ranked"] if not is_combined_record(record)]
    if reference_rows:
        with st.expander("Reference runs (original, noise-only and contrast-only)"):
            st.dataframe(
                [
                    {
                        "ID": record.get("id", "—"),
                        "Module": module_label(record.get("module")),
                        "Technique": technique_label(record.get("technique")),
                        "Stage": (
                            "Original" if record.get("technique") == "original" else
                            "Noise-only" if record.get("technique") in {value[0] for value in MEMBER_TECHNIQUES.values()} else
                            "Contrast-only" if record.get("technique") in {value[1] for value in MEMBER_TECHNIQUES.values()} else
                            "Other reference"
                        ),
                        "Split": record.get("split", "—"),
                        "mAP50-95": round(_metric_value(record, "map50_95"), 4),
                    }
                    for record in reference_rows
                ],
                width="stretch",
                hide_index=True,
            )

def _render_reproducibility(st, selected: dict, model_id: str, *, key_prefix: str) -> None:
    checkpoint = _checkpoint_for_model(model_id)
    manifest = {
        "experiment": selected.get("id"),
        "model": model_id,
        "checkpoint": str(checkpoint),
        "module": selected.get("module"),
        "technique": selected.get("technique"),
        "parameters": selected.get("parameters", {}),
        "protocol": _protocol_payload(),
    }
    st.subheader("Reproducibility")
    st.caption("Exact settings used for the selected model and preprocessing preset.")
    st.json(manifest)
    st.download_button(
        "Download selected experiment config",
        data=dumps_json(manifest),
        file_name=f"{selected.get('id', 'experiment')}_config.json",
        mime="application/json",
        key=f"{key_prefix}_download_selected_config",
    )


@__import__("streamlit").cache_resource
def _load_model(checkpoint: str):
    from ultralytics import YOLO

    return YOLO(checkpoint)


def _decode_upload(uploaded_file) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {uploaded_file.name}")
    return image


def _decode_payload(name: str, payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {name}")
    return image


def _resize_for_inference(image: np.ndarray) -> np.ndarray:
    """Keep paired inference and preprocessing within the hosted app memory budget."""

    longest_side = max(image.shape[:2])
    if longest_side <= INFERENCE_MAX_SIDE:
        return image.copy()
    scale = INFERENCE_MAX_SIDE / longest_side
    size = (
        max(1, round(image.shape[1] * scale)),
        max(1, round(image.shape[0] * scale)),
    )
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _update_progress(progress, done: int, total: int, label: str) -> None:
    """Show a bounded progress fraction and a human-readable batch position."""

    if total > 0:
        fraction = min(max(done / total, 0.0), 1.0)
        progress.progress(fraction, text=f"Processing {label} {done}/{total}")
    else:
        progress.progress(0.0, text=f"Processing {label} {done}")


def _image_upload_key(st) -> str:
    version = int(st.session_state.get(IMAGE_UPLOAD_VERSION_KEY, 0))
    return f"image_inference_uploads_{version}"


def _clear_image_uploads(st) -> None:
    current_version = int(st.session_state.get(IMAGE_UPLOAD_VERSION_KEY, 0))
    st.session_state[IMAGE_UPLOAD_VERSION_KEY] = current_version + 1


def _detect(model, image: np.ndarray) -> tuple[np.ndarray, int]:
    prediction_stream = model.predict(
        source=image,
        imgsz=INFERENCE_IMGSZ,
        conf=INFERENCE_CONF,
        iou=INFERENCE_IOU,
        device=select_device("auto"),
        verbose=False,
        stream=True,
    )
    result = next(iter(prediction_stream))
    plotted = result.plot()
    count = int(len(result.boxes)) if result.boxes is not None else 0
    output = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    del prediction_stream, result, plotted
    return output, count


def _run_inference_pair(model, original: np.ndarray, selected: dict) -> dict:
    """Run the same model on the original and preprocessed versions of an image."""

    working = _resize_for_inference(original)
    original_model_result, original_model_detections = _detect(model, working)
    processed = apply_candidate(working, _candidate_from_record(selected))
    result, preprocessed_detections = _detect(model, processed)
    return {
        "processed": processed,
        "original_model_result": original_model_result,
        "original_model_detections": original_model_detections,
        "result": result,
        "preprocessed_detections": preprocessed_detections,
    }


def _render_inference_visual_result(st, item: dict) -> None:
    """Render original input and both same-model detection comparisons."""

    c1, c2, c3, c4 = st.columns(4)
    c1.image(item["original"], caption="Uploaded original", width="stretch")
    c2.image(item["original_model_result"], caption="Original model detection", width="stretch")
    c3.image(item["processed"], caption="After selected preprocessing", width="stretch")
    c4.image(item["result"], caption="Preprocessed detection result", width="stretch")


def _option_label(value: str) -> str:
    return "All" if value == "all" else technique_label(value)


def _module_option_label(value: str) -> str:
    return "All" if value == "all" else module_label(value)


def _select_value(st, label: str, values: list[str], *, key: str, format_func=_option_label) -> str:
    choices = ["all", *values]
    current = st.session_state.get(key, "all")
    if current not in choices:
        current = "all"
        st.session_state[key] = current
    kwargs = {"format_func": format_func, "key": key}
    if key not in st.session_state:
        kwargs["index"] = choices.index(current)
    return st.selectbox(label, choices, **kwargs)


# The Technique dropdown is split into "Filtering" (noise removal) and
# "Contrast" (enhancement) pickers. Both stages come from the same existing
# per-module technique pairs already used by the analysis module, so this is
# purely a presentation split: no new technique values are introduced and no
# preprocessing code is touched. Either side can be left as "Don't apply".
DONT_APPLY = "dont_apply"
NOISE_TECHNIQUES = {pair[0] for pair in MEMBER_TECHNIQUES.values()}
CONTRAST_TECHNIQUES = {pair[1] for pair in MEMBER_TECHNIQUES.values()}
_MODULE_BY_NOISE = {pair[0]: module for module, pair in MEMBER_TECHNIQUES.items()}
_MODULE_BY_CONTRAST = {pair[1]: module for module, pair in MEMBER_TECHNIQUES.items()}


def _decompose_technique(technique: str) -> tuple[str, str]:
    """Split an existing technique value into its (filtering, contrast) parts."""

    if technique in NOISE_TECHNIQUES:
        return technique, DONT_APPLY
    if technique in CONTRAST_TECHNIQUES:
        return DONT_APPLY, technique
    for noise, contrast in MEMBER_TECHNIQUES.values():
        if technique == f"{noise}_{contrast}":
            return noise, contrast
    return DONT_APPLY, DONT_APPLY


def _select_optional(st, label: str, values: list[str], *, key: str) -> str:
    choices = [DONT_APPLY, *values]
    current = st.session_state.get(key, DONT_APPLY)
    if current not in choices:
        current = DONT_APPLY
        st.session_state[key] = current
    format_func = lambda value: "Don't apply" if value == DONT_APPLY else technique_label(value)
    kwargs = {"format_func": format_func, "key": key}
    if key not in st.session_state:
        kwargs["index"] = choices.index(current)
    return st.selectbox(label, choices, **kwargs)


def _render_comparison_filters(st, records: list[dict]) -> dict[str, str]:
    keys = {field: f"compare_{field}" for field in FILTER_FIELDS}
    selection = {
        field: st.session_state.get(key, "all")
        for field, key in keys.items()
    }
    selection = normalize_selection(records, selection)
    for field, key in keys.items():
        st.session_state[key] = selection[field]

    model_col, split_col, module_col, technique_col = st.columns(4)
    with model_col:
        selection["model"] = _select_value(
            st, "Model", option_values(records)["model"], key=keys["model"]
        )
    selection = normalize_selection(records, selection)
    with split_col:
        split_options = option_values(records, model=selection["model"])["split"]
        selection["split"] = _select_value(
            st, "Dataset split", split_options, key=keys["split"]
        )
    selection = normalize_selection(records, selection)
    with module_col:
        module_options = option_values(
            records, model=selection["model"], split=selection["split"]
        )["module"]
        selection["module"] = _select_value(
            st, "Module", module_options, key=keys["module"], format_func=_module_option_label
        )
    selection = normalize_selection(records, selection)
    with technique_col:
        technique_options = option_values(
            records,
            model=selection["model"],
            split=selection["split"],
            module=selection["module"],
        )["technique"]
        selection["technique"] = _select_value(
            st, "Technique", technique_options, key=keys["technique"]
        )
    selection = normalize_selection(records, selection)
    run_type_options = ["all", "combined", "reference"]
    run_type_labels = {
        "combined": "Combined + original baseline",
        "reference": "Reference runs",
        "all": "All runs",
    }
    current_run_type = st.session_state.get("compare_run_type", "all")
    if current_run_type not in run_type_options:
        current_run_type = "all"
    selection["run_type"] = st.selectbox(
        "Run type",
        run_type_options,
        index=run_type_options.index(current_run_type),
        format_func=run_type_labels.get,
        key="compare_run_type",
    )
    return selection


def _render_inference_filters(st, records: list[dict], *, key_prefix: str = "infer") -> dict[str, str]:
    model_key, module_key, technique_key = inference_widget_keys(key_prefix)
    selection = {
        "model": st.session_state.get(model_key, "all"),
        "split": "all",
        "module": st.session_state.get(module_key, "all"),
        "technique": st.session_state.get(technique_key, "all"),
    }
    selection = normalize_selection(records, selection)
    st.session_state[model_key] = selection["model"]
    st.session_state[module_key] = selection["module"]
    st.session_state[technique_key] = selection["technique"]

    model_col, module_col, filtering_col, contrast_col = st.columns(4)
    with model_col:
        selection["model"] = _select_value(
            st, "Model", option_values(records)["model"], key=model_key
        )
    selection = normalize_selection(records, selection)
    with module_col:
        module_options = option_values(records, model=selection["model"])["module"]
        selection["module"] = _select_value(
            st, "Module", module_options, key=module_key
        )
    selection = normalize_selection(records, selection)

    technique_options = option_values(
        records, model=selection["model"], module=selection["module"]
    )["technique"]

    # Filtering and Contrast are two views onto the same "technique" selection,
    # so keep them in sync with technique_key rather than owning state of
    # their own: resync from technique_key whenever it changed outside these
    # two widgets (e.g. the "Use recommended experiment" button below).
    filtering_key, contrast_key = f"{key_prefix}_filtering", f"{key_prefix}_contrast"
    sync_key = f"{key_prefix}_technique_sync"
    if st.session_state.get(sync_key) != selection["technique"]:
        filtering_default, contrast_default = _decompose_technique(selection["technique"])
        st.session_state[filtering_key] = filtering_default
        st.session_state[contrast_key] = contrast_default

    with filtering_col:
        filtering_options = sorted(NOISE_TECHNIQUES & set(technique_options))
        filtering_choice = _select_optional(st, "Filtering", filtering_options, key=filtering_key)
    with contrast_col:
        contrast_options = sorted(CONTRAST_TECHNIQUES & set(technique_options))
        if filtering_choice != DONT_APPLY:
            paired_module = _MODULE_BY_NOISE.get(filtering_choice)
            contrast_options = [
                value for value in contrast_options if _MODULE_BY_CONTRAST.get(value) == paired_module
            ]
        contrast_choice = _select_optional(st, "Contrast", contrast_options, key=contrast_key)

    if filtering_choice == DONT_APPLY and contrast_choice == DONT_APPLY:
        selection["technique"] = "all"
    elif contrast_choice == DONT_APPLY:
        selection["technique"] = filtering_choice
    elif filtering_choice == DONT_APPLY:
        selection["technique"] = contrast_choice
    else:
        selection["technique"] = f"{filtering_choice}_{contrast_choice}"
    if selection["technique"] not in {"all", *technique_options}:
        selection["technique"] = "all"

    st.session_state[technique_key] = selection["technique"]
    st.session_state[sync_key] = selection["technique"]
    selection = normalize_selection(records, selection)
    return selection


def _render_active_experiment(st, record: dict, *, heading: str = "Active experiment") -> None:
    st.subheader(heading)
    st.caption(
        f"{record.get('model_label', record.get('model_id', 'baseline'))} · "
        f"{module_label(record.get('module'))} · {technique_label(record.get('technique'))} · {record.get('id', '—')}"
    )
    cards = st.columns(5)
    cards[0].metric("Image size", str(INFERENCE_IMGSZ))
    cards[1].metric("Confidence", f"{INFERENCE_CONF:.2f}")
    cards[2].metric("IoU", f"{INFERENCE_IOU:.2f}")
    cards[3].metric("Split", str(record.get("split", "—")))
    cards[4].metric("mAP50-95", f"{float(record.get('metrics', {}).get('map50_95', 0)):.4f}")
    with st.expander("Exact preprocessing parameters", expanded=True):
        parameters = dict(record.get("parameters") or {})
        if record.get("module") == "member5":
            # Keep every Member 5 control visible even when a malformed or
            # legacy record omitted one, while retaining any future extras.
            parameters = {
                key: parameters.get(key)
                for key in MEMBER5_PARAMETER_KEYS
            } | {
                key: value
                for key, value in parameters.items()
                if key not in MEMBER5_PARAMETER_KEYS
            }
        st.json(parameters)


def _render_recommendation_extras(st, records: list[dict]) -> None:
    """Member 2's required-combo note and the full best-by-module table.

    The headline recommendation (id, score, model/module/technique, and the
    "use recommended preset" action) now lives in each page's own Preset
    card; this covers the remaining detail that card doesn't show.
    """

    member2 = next(
        (row for row in best_by_module(records) if row.get("module") == "member2"),
        None,
    )
    if member2:
        parameters = member2.get("parameters", {})
        level = parameters.get("wavelet_levels")
        st.info(
            "Member 2 final assignment preset (Wavelet → Homomorphic): "
            f"{parameters.get('wavelet_name')} / {parameters.get('wavelet_method')} / "
            f"{parameters.get('wavelet_mode')} / level {'auto' if level is None else level}; "
            f"γL={parameters.get('homomorphic_gamma_low')}, "
            f"γH={parameters.get('homomorphic_gamma_high')}, "
            f"cutoff={parameters.get('homomorphic_cutoff')}, "
            f"sharpness={parameters.get('homomorphic_sharpness')}; "
            f"validation mAP50-95={_metric_value(member2, 'map50_95'):.4f}."
        )

    module_rows = best_by_module(records)
    if module_rows:
        with st.expander("Best experiment by module"):
            st.dataframe(
                [
                    {
                        "Module": module_label(row.get("module")),
                        "Technique": technique_label(row.get("technique")),
                        "Parameters": json.dumps(row.get("parameters", {}), sort_keys=True),
                        "mAP50-95": round(float(row.get("metrics", {}).get("map50_95", 0)), 4),
                        "Experiment": row.get("id", "—"),
                    }
                    for row in module_rows
                ],
                width="stretch",
                hide_index=True,
            )


def _render_comparison_mode(st, records: list[dict], results_path: Path) -> None:
    st.header("Compare experiments")
    st.caption("Best run and default ranking use the four member combined techniques. Reference runs remain available for comparison.")
    selection = _render_comparison_filters(st, records)
    st.info("Use val for tuning and comparison. Use test only for the final frozen comparison.")
    sort_col, direction_col, reset_col = st.columns([2, 1.5, 1])
    metrics = sorted({key for record in records for key in record.get("metrics", {})})
    default_index = metrics.index("map50_95") if "map50_95" in metrics else 0
    with sort_col:
        sort_metric = st.selectbox(
            "Sort by",
            metrics or ["map50_95"],
            index=default_index,
            format_func=_label,
            key="compare_sort_metric",
        )
    with direction_col:
        direction = st.radio(
            "Order", ["High to low", "Low to high"], horizontal=True, key="compare_direction"
        )
    with reset_col:
        st.write("")
        st.button(
            "Reset filters",
            key="compare_reset",
            on_click=reset_selection_state,
            kwargs={
                "state": st.session_state,
                "prefix": "compare_",
                "extra_fields": ("run_type",),
                "defaults": {"run_type": "all"},
            },
        )

    filtered = comparison_records(records, **selection)
    filtered.sort(
        key=lambda record: float(record.get("metrics", {}).get(sort_metric, float("-inf"))),
        reverse=direction == "High to low",
    )
    if not filtered:
        st.warning(
            "No records match: "
            + " / ".join(f"{field}={selection[field]}" for field in FILTER_FIELDS)
        )
        return

    best_combined = best_experiment(records)
    cards = st.columns(4)
    cards[0].metric("Visible runs", len(filtered))
    cards[1].metric(_label(sort_metric), f"{float(filtered[0].get('metrics', {}).get(sort_metric, 0)):.4f}")
    cards[2].metric(
        "Best combined mAP50-95",
        f"{_metric_value(best_combined, 'map50_95'):.4f}" if best_combined else "—",
    )
    cards[3].metric("Combined modules", len({record.get("module") for record in filtered if is_combined_record(record)}))
    if best_combined:
        st.success(
            f"Best combined run: {best_combined.get('id', '—')} · "
            f"{module_label(best_combined.get('module'))} / {technique_label(best_combined.get('technique'))}"
        )
        st.caption(
            "Parameters: "
            + json.dumps(best_combined.get("parameters", {}), sort_keys=True)
        )

    chart_rows, baseline_value = ranking_chart_rows(filtered, sort_metric)
    if chart_rows:
        st.subheader(f"Ranking by {_label(sort_metric)}")
        if baseline_value is None:
            st.caption("No unprocessed control is present in the current filter, so no baseline line is drawn.")
        else:
            st.caption(f"The vertical line marks the unprocessed baseline at {baseline_value:.4f}.")
        bars = (
            alt.Chart(alt.Data(values=chart_rows))
            .mark_bar(cornerRadiusEnd=4, height=17)
            .encode(
                x=alt.X("value:Q", title=_label(sort_metric)),
                y=alt.Y("label:N", sort="-x", title=None),
                color=alt.Color(
                    "is_extra:N",
                    title=None,
                    scale=alt.Scale(domain=[False, True], range=["#2563eb", "#a9b6c6"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("member:N", title="Member"),
                    alt.Tooltip("technique:N", title="Technique"),
                    alt.Tooltip("value:Q", title=_label(sort_metric), format=".4f"),
                ],
            )
        )
        layers = [bars]
        if baseline_value is not None:
            layers.append(
                alt.Chart(alt.Data(values=[{"baseline": baseline_value}]))
                .mark_rule(color="#172033", strokeWidth=2)
                .encode(x="baseline:Q")
            )
        st.altair_chart(alt.layer(*layers).properties(height=max(120, 26 * len(chart_rows))), use_container_width=True)

    table = []
    for record in filtered:
        row = {
            "ID": record["id"],
            "Model": record.get("model_id", "baseline"),
            "Module": "Baseline control" if record.get("id") == "original_shared_control" else module_label(record.get("module")),
            "Technique": record.get("display_label", technique_label(record.get("technique"))),
            "Split": record.get("split", "—"),
        }
        row.update({_label(key): value for key, value in record.get("parameters", {}).items()})
        row.update({
            _label(key): round(float(value), 4) if isinstance(value, (int, float)) else value
            for key, value in record.get("metrics", {}).items()
        })
        table.append(row)
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download filtered JSON",
        dumps_json(filtered),
        file_name="filtered_results.json",
        mime="application/json",
        key="compare_download",
    )

    selected_ids = [record["id"] for record in filtered]
    selected_id = st.selectbox("Inspect experiment", selected_ids, key="compare_selected_id")
    selected = next(record for record in filtered if record["id"] == selected_id)
    _render_active_experiment(st, selected, heading="Selected experiment")
    left, right = st.columns([1.1, 1])
    preview_path = _resolve_preview_path(results_path, selected.get("preview"))
    with left:
        st.subheader(selected["id"])
        if preview_path is not None:
            st.image(
                str(preview_path),
                caption=f"{module_label(selected.get('module'))} / {technique_label(selected.get('technique'))}",
                width="stretch",
            )
        else:
            st.info(
                "Preview image is not included in this deployment. "
                "Metrics and parameters are still available."
            )
    with right:
        st.subheader("Parameters and metrics")
        st.json({
            "model": selected.get("model_label", selected.get("model_id", "baseline")),
            "split": selected.get("split"),
            "training_preprocessing": selected.get("training_preprocessing"),
            "evaluation_preprocessing": selected.get("evaluation_preprocessing"),
            "parameters": selected.get("parameters", {}),
            "metrics": selected.get("metrics", {}),
        })


def _render_page_header(st, *, page: str, title: str, subtitle: str) -> None:
    """Breadcrumb + title + subtitle shared by the workbench-style pages."""

    st.markdown(
        "<div style='color:#8a97ab;font-size:0.85rem;margin-bottom:0.5rem;'>"
        "&#8598; Workspace &nbsp;&rsaquo;&nbsp; "
        f"<span style='color:#dc2626;font-weight:600;'>{page}</span></div>",
        unsafe_allow_html=True,
    )
    st.title(title)
    st.caption(subtitle)


def _render_step_indicator(st, steps: list[str], current_index: int) -> None:
    """A decorative horizontal stepper: done / active / upcoming."""

    parts = []
    for index, label in enumerate(steps):
        if index < current_index:
            circle_style, text_color = "background:#059669;color:white;", "#059669"
        elif index == current_index:
            circle_style, text_color = "background:#2563eb;color:white;", "#172033"
        else:
            circle_style, text_color = "background:white;color:#8a97ab;border:1px solid #d7deea;", "#8a97ab"
        parts.append(
            "<div style='display:flex;align-items:center;gap:0.5rem;'>"
            "<div style='width:1.6rem;height:1.6rem;border-radius:50%;display:flex;"
            f"align-items:center;justify-content:center;font-size:0.8rem;font-weight:600;{circle_style}'>"
            f"{index + 1}</div><span style='color:{text_color};font-weight:600;font-size:0.9rem;'>"
            f"{label}</span></div>"
        )
        if index != len(steps) - 1:
            parts.append("<div style='flex:1;height:1px;background:#d7deea;margin:0 0.9rem;'></div>")
    st.markdown(
        "<div style='display:flex;align-items:center;padding:1rem 1.25rem;background:white;"
        "border:1px solid #e3e9f2;border-radius:14px;margin-bottom:1.1rem;'>" + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def _render_scroll_anchor(st, anchor_id: str) -> None:
    """Mark a spot in the page and scroll it into view on this run only.

    Used right above a results section so the page jumps to it once
    processing finishes, without affecting any later, unrelated rerun.
    """

    st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
    st.components.v1.html(
        f"""
        <script>
        const target = window.parent.document.getElementById("{anchor_id}");
        if (target) {{
            target.scrollIntoView({{behavior: "smooth", block: "start"}});
        }}
        </script>
        """,
        height=0,
    )


def _render_inference_mode(st, records: list[dict]) -> None:
    _render_page_header(
        st,
        page=NAV_IMAGE_INFERENCE,
        title="Image processing workbench",
        subtitle="Evaluate a preprocessing preset on real PCB images.",
    )

    uploads_key = _image_upload_key(st)
    has_uploads = bool(st.session_state.get(uploads_key))
    _render_step_indicator(
        st, ["Select preset", "Upload images", "Review results"], 1 if has_uploads else 0
    )

    recommended = best_experiment(records)
    preset_col, upload_col = st.columns(2)

    with preset_col, st.container(border=True):
        header_col, badge_col = st.columns([3, 2])
        header_col.subheader("Preset")

        selection = _render_inference_filters(st, records)
        candidates = filter_records(
            records, model=selection["model"], module=selection["module"], technique=selection["technique"],
        )
        if not candidates:
            st.warning("No inference preset matches the selected model, module, and technique.")
            return
        experiment_ids = [record["id"] for record in candidates]
        current_id = st.session_state.get("infer_experiment", experiment_ids[0])
        if current_id not in experiment_ids:
            current_id = experiment_ids[0]
            st.session_state["infer_experiment"] = current_id
        experiment_kwargs = {"key": "infer_experiment"}
        if "infer_experiment" not in st.session_state:
            experiment_kwargs["index"] = experiment_ids.index(current_id)
        selected_id = st.selectbox("Experiment", experiment_ids, **experiment_kwargs)
        selected = next(record for record in candidates if record["id"] == selected_id)

        if recommended is not None and selected_id == recommended.get("id"):
            badge_col.markdown(
                "<div style='text-align:right;padding-top:0.4rem;'><span style='background:#e7f6ee;"
                "color:#059669;padding:0.25rem 0.65rem;border-radius:999px;font-size:0.8rem;"
                "font-weight:600;'>&#127942; Recommended (best combined)</span></div>",
                unsafe_allow_html=True,
            )

        if recommended is not None:
            with st.container(border=True):
                info_col, score_col = st.columns([3, 1])
                info_col.caption("Best combined experiment")
                info_col.markdown(f"**{recommended.get('id', '—')}**")
                score_col.metric("mAP50-95", f"{_metric_value(recommended, 'map50_95'):.4f}")

        detail_cols = st.columns(3)
        detail_cols[0].caption("Model")
        detail_cols[0].markdown(f"**{selected.get('model_id', 'baseline')}**")
        detail_cols[1].caption("Module")
        detail_cols[1].markdown(f"**{module_label(selected.get('module'))}**")
        detail_cols[2].caption("Technique")
        detail_cols[2].markdown(f"**{technique_label(selected.get('technique'))}**")

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            if recommended is not None and st.button(
                "Use recommended preset", key="use_recommended_infer", type="primary", width="stretch"
            ):
                model_key, module_key, technique_key = inference_widget_keys("infer")
                st.session_state[model_key] = recommended.get("model_id", "baseline")
                st.session_state[module_key] = recommended.get("module", "all")
                st.session_state[technique_key] = recommended.get("technique", "all")
                st.session_state["infer_experiment"] = recommended.get("id")
                st.rerun()
        with button_col2:
            show_params = st.session_state.get("infer_show_params", False)
            if st.button(
                "Hide exact preprocessing parameters" if show_params else "Show exact preprocessing parameters",
                key="infer_toggle_params",
                width="stretch",
            ):
                st.session_state["infer_show_params"] = not show_params
                st.rerun()

    with upload_col, st.container(border=True):
        st.subheader("Upload PCB images")
        uploads = st.file_uploader(
            "Upload PCB images or one ZIP folder",
            type=["jpg", "jpeg", "png", "zip"],
            accept_multiple_files=True,
            key=uploads_key,
            label_visibility="collapsed",
        )
        st.caption("Supported formats: JPG, PNG, ZIP")
        run_clicked = st.button(
            "Run detection", key="run_inference", type="primary", width="stretch", disabled=not uploads
        )
        if uploads and st.button("Clear all", key="clear_image_uploads", width="stretch"):
            _clear_image_uploads(st)
            st.rerun()

    _render_recommendation_extras(st, records)

    if st.session_state.get("infer_show_params", False):
        _render_active_experiment(st, selected)
        _render_reproducibility(st, selected, selected.get("model_id", "baseline"), key_prefix="infer")

    selected_model = selected.get("model_id", "baseline")
    selected_checkpoint = _checkpoint_for_model(selected_model)
    if not selected_checkpoint.is_file():
        st.error(f"Checkpoint not available for {selected_model}: {selected_checkpoint}")
        return

    if run_clicked and uploads:
        image_entries, skipped = extract_image_entries(
            [(upload.name, upload.getvalue()) for upload in uploads]
        )
        if skipped:
            st.warning("Some files were skipped:\n\n" + "\n".join(f"- {message}" for message in skipped))
        st.info(f"Prepared {len(image_entries)} image(s) for detection; skipped {len(skipped)} file(s).")
        total_images = len(image_entries)
        progress = st.progress(0.0, text=f"Loading YOLO model for {total_images} image(s)...")
        model = _load_model(str(selected_checkpoint))
        progress.progress(0.0, text=f"Processing image 0/{total_images}")
        started = time.perf_counter()
        summary = []
        visual_results = []
        for index, (filename, payload) in enumerate(image_entries, start=1):
            try:
                original = _decode_payload(filename, payload)
                inference = _run_inference_pair(model, original, selected)
            except (ValueError, cv2.error) as error:
                st.error(f"{filename}: {error}")
            else:
                summary.append({
                    "file": filename,
                    "original_model_detections": inference["original_model_detections"],
                    "preprocessed_detections": inference["preprocessed_detections"],
                    "detections": inference["preprocessed_detections"],
                    "model": selected_model,
                    "experiment": selected["id"],
                })
                visual_results.append({
                    "file": filename,
                    "original_model_detections": inference["original_model_detections"],
                    "preprocessed_detections": inference["preprocessed_detections"],
                    "original": cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                    "processed": cv2.cvtColor(inference["processed"], cv2.COLOR_BGR2RGB),
                    "original_model_result": inference["original_model_result"],
                    "result": inference["result"],
                })
            finally:
                _update_progress(progress, index, total_images, "image")
        elapsed = time.perf_counter() - started
        progress.progress(1.0, text=f"Image detection complete: {len(summary)}/{total_images} succeeded")

        _render_step_indicator(st, ["Select preset", "Upload images", "Review results"], 2)
        _render_scroll_anchor(st, "infer-results")
        header_col, download_col = st.columns([4, 1])
        header_col.subheader("Image references")
        if summary:
            download_col.download_button(
                "Download summary (JSON)",
                dumps_json(summary),
                file_name="inference_summary.json",
                mime="application/json",
                key="inference_download",
            )
        metric_cols = st.columns(4)
        metric_cols[0].metric("mAP50-95", f"{_metric_value(selected, 'map50_95'):.4f}")
        metric_cols[1].metric("Original detections", sum(row["original_model_detections"] for row in summary))
        metric_cols[2].metric("Preprocessed detections", sum(row["preprocessed_detections"] for row in summary))
        metric_cols[3].metric("Processing time", f"{elapsed:.1f} s")
        if summary:
            st.dataframe(summary, width="stretch", hide_index=True)
        for index, item in enumerate(visual_results):
            with st.expander(
                f"{item['file']} - original model: {item['original_model_detections']} detections; "
                f"preprocessed: {item['preprocessed_detections']} detections",
                expanded=index == 0,
            ):
                _render_inference_visual_result(st, item)


def _render_video_mode(st, records: list[dict]) -> None:
    _render_page_header(
        st,
        page=NAV_VIDEO,
        title="Video processing",
        subtitle="Apply a preprocessing preset frame by frame and review the annotated output.",
    )

    has_upload = bool(st.session_state.get("video_inference_upload"))
    _render_step_indicator(
        st, ["Choose preset", "Upload video", "Process", "Export"], 1 if has_upload else 0
    )

    recommended = best_experiment(records)
    preset_col, upload_col = st.columns(2)

    with preset_col, st.container(border=True):
        header_col, badge_col = st.columns([3, 2])
        header_col.subheader("Processing preset")

        selection = _render_inference_filters(st, records, key_prefix="video")
        candidates = filter_records(
            records, model=selection["model"], module=selection["module"], technique=selection["technique"],
        )
        if not candidates:
            st.warning("No video preset matches the selected model, module, and technique.")
            return
        experiment_ids = [record["id"] for record in candidates]
        current_id = st.session_state.get("video_experiment", experiment_ids[0])
        if current_id not in experiment_ids:
            current_id = experiment_ids[0]
        selected_id = st.selectbox(
            "Experiment", experiment_ids, index=experiment_ids.index(current_id), key="video_experiment"
        )
        selected = next(record for record in candidates if record["id"] == selected_id)

        if recommended is not None and selected_id == recommended.get("id"):
            badge_col.markdown(
                "<div style='text-align:right;padding-top:0.4rem;'><span style='background:#e7f6ee;"
                "color:#059669;padding:0.25rem 0.65rem;border-radius:999px;font-size:0.8rem;"
                "font-weight:600;'>&#127942; Recommended (best combined)</span></div>",
                unsafe_allow_html=True,
            )

        if recommended is not None:
            with st.container(border=True):
                info_col, score_col = st.columns([3, 1])
                info_col.caption("Recommended best combined experiment")
                info_col.markdown(f"**{recommended.get('id', '—')}**")
                score_col.metric("mAP50-95", f"{_metric_value(recommended, 'map50_95'):.4f}")

        detail_cols = st.columns(3)
        detail_cols[0].caption("Model")
        detail_cols[0].markdown(f"**{selected.get('model_id', 'baseline')}**")
        detail_cols[1].caption("Module")
        detail_cols[1].markdown(f"**{module_label(selected.get('module'))}**")
        detail_cols[2].caption("Technique")
        detail_cols[2].markdown(f"**{technique_label(selected.get('technique'))}**")

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            if recommended is not None and st.button(
                "Use recommended preset", key="use_recommended_video", type="primary", width="stretch"
            ):
                model_key, module_key, technique_key = inference_widget_keys("video")
                st.session_state[model_key] = recommended.get("model_id", "baseline")
                st.session_state[module_key] = recommended.get("module", "all")
                st.session_state[technique_key] = recommended.get("technique", "all")
                st.session_state["video_experiment"] = recommended.get("id")
                st.rerun()
        with button_col2:
            show_params = st.session_state.get("video_show_params", False)
            if st.button(
                "Hide exact preprocessing parameters" if show_params else "View exact preprocessing parameters",
                key="video_toggle_params",
                width="stretch",
            ):
                st.session_state["video_show_params"] = not show_params
                st.rerun()

    with upload_col, st.container(border=True):
        st.subheader("Upload a short video")
        st.caption("Upload a short video (recommended ≤ 60 seconds) to process.")
        video = st.file_uploader(
            "Upload a short video",
            type=["mp4", "mov", "avi"],
            accept_multiple_files=False,
            key="video_inference_upload",
            label_visibility="collapsed",
        )
        st.caption("Supports MP4, MOV, AVI")
        run_clicked = st.button(
            "Process video", key="run_video_detection", type="primary", width="stretch", disabled=not video
        )

    _render_recommendation_extras(st, records)

    if st.session_state.get("video_show_params", False):
        _render_active_experiment(st, selected, heading="Selected video experiment")
        _render_reproducibility(st, selected, selected.get("model_id", "baseline"), key_prefix="video")

    selected_model = selected.get("model_id", "baseline")
    checkpoint = _checkpoint_for_model(selected_model)
    if not checkpoint.is_file():
        st.error(f"Checkpoint not available for {selected_model}: {checkpoint}")
        return
    if run_clicked and video:
        _render_step_indicator(st, ["Choose preset", "Upload video", "Process", "Export"], 2)
        progress = st.progress(0.0, text="Loading YOLO model...")
        try:
            model = _load_model(str(checkpoint))
            progress.progress(0.0, text="Preparing video...")
            with tempfile.TemporaryDirectory(prefix="hripcb_video_") as temp_dir:
                input_path = Path(temp_dir) / video.name
                output_path = Path(temp_dir) / "annotated_output.mp4"
                input_path.write_bytes(video.getvalue())

                def update_progress(done: int, total: int) -> None:
                    _update_progress(progress, done, total, "frame")

                summary = process_video(
                    input_path,
                    output_path,
                    model,
                    _candidate_from_record(selected),
                    imgsz=INFERENCE_IMGSZ,
                    conf=INFERENCE_CONF,
                    iou=INFERENCE_IOU,
                    device=select_device("auto"),
                    progress_callback=update_progress,
                )
                progress.progress(0.99, text="Finalizing video preview...")
                output_bytes = output_path.read_bytes()
            progress.progress(1.0, text="Video processing complete")
        except (ValueError, cv2.error) as error:
            progress.empty()
            st.error(str(error))
            return
        _render_step_indicator(st, ["Choose preset", "Upload video", "Process", "Export"], 3)
        _render_scroll_anchor(st, "video-results")
        st.subheader("Output preview")
        cards = st.columns(4)
        cards[0].metric("Frames", summary["frames"])
        cards[1].metric("FPS", f"{summary['fps']:.2f}")
        cards[2].metric("Detections", summary["detections"])
        cards[3].metric("Resolution", f"{summary['width']} × {summary['height']}")
        if summary.get("browser_compatible"):
            st.video(output_bytes)
        else:
            st.warning(
                "Detection completed, but this environment could not create a browser-compatible preview. "
                "Download the annotated video to view it."
            )
        st.download_button(
            "Download annotated video",
            data=output_bytes,
            file_name="hripcb_annotated_output.mp4",
            mime="video/mp4",
            key="download_annotated_video",
        )
        st.download_button(
            "Download video processing summary",
            data=dumps_json({"experiment": selected["id"], "protocol": _protocol_payload(), "summary": summary}),
            file_name="hripcb_video_summary.json",
            mime="application/json",
            key="download_video_summary",
        )


# Left-hand navigation pages. Each maps straight onto an existing, unmodified
# render function -- this only changes how the user reaches them, not what
# they do or how any preprocessing/detection pipeline runs.
NAV_DASHBOARD = "Dashboard"
NAV_EXPERIMENTS = "Experiments"
NAV_IMAGE_INFERENCE = "Image processing"
NAV_ANALYSIS = "Analysis & reports"
NAV_VIDEO = "Video processing"
NAV_PAGES = (NAV_DASHBOARD, NAV_EXPERIMENTS, NAV_IMAGE_INFERENCE, NAV_ANALYSIS, NAV_VIDEO)
NAV_STATE_KEY = "nav_page"

# Fixed by the frozen protocol (see README section 4) -- not derived from the
# records, so shown as static project context rather than computed metrics.
PROJECT_CONTEXT = {"Dataset": "HRIPCB_UPDATE", "Model": "YOLOv8s", "Split": "validation"}


def _go_to(st, page: str) -> None:
    st.session_state[NAV_STATE_KEY] = page
    st.rerun()


def _render_sidebar_nav(st) -> str:
    current = st.session_state.get(NAV_STATE_KEY, NAV_DASHBOARD)
    if current not in NAV_PAGES:
        current = NAV_DASHBOARD
    with st.sidebar:
        logo_path = PROJECT_ROOT / "assets/logo.png"
        if logo_path.is_file():
            st.image(str(logo_path), width="stretch")
        else:
            st.markdown("### 🔬 HRIPCB Lab")
        st.caption("A shared, report-ready workspace for Member 1–5 preprocessing experiments.")
        st.caption("WORKSPACE")
        with st.container(key="nav_list"):
            for page in NAV_PAGES:
                is_active = page == current
                if st.button(
                    page,
                    key=f"nav_{page}",
                    type="primary" if is_active else "secondary",
                    width="stretch",
                ) and not is_active:
                    _go_to(st, page)
        st.divider()
        st.caption("PROJECT CONTEXT")
        for label, value in PROJECT_CONTEXT.items():
            st.markdown(f"**{label}**  \n{value}")
    return current


def _render_dashboard_home(st, records: list[dict]) -> None:
    import pandas as pd

    st.title("Dashboard")
    st.caption("A clear starting point for running experiments, reviewing evidence, and exporting results.")

    summary = record_metric_summary(records)
    best = summary["best"]
    cards = st.columns(4)
    cards[0].metric("Total runs", summary["count"])
    cards[1].metric("Combined runs", summary["combined_count"])
    cards[2].metric("Modules", summary["module_count"])
    cards[3].metric("Best mAP50-95", f"{_metric_value(best, 'map50_95'):.4f}" if best else "—")

    st.subheader("Best combined result")
    if best is None:
        st.warning("No combined validation result is available yet.")
    else:
        with st.container(border=True):
            info_col, action_col = st.columns([3, 1])
            with info_col:
                detail_cols = st.columns(4)
                detail_cols[0].markdown(f"**Experiment**  \n`{best.get('id', '—')}`")
                detail_cols[1].markdown(f"**Module**  \n{module_label(best.get('module'))}")
                detail_cols[2].markdown(f"**Technique**  \n{technique_label(best.get('technique'))}")
                detail_cols[3].markdown(f"**mAP50-95**  \n{_metric_value(best, 'map50_95'):.4f}")
                st.caption(f"Validated on: {best.get('split', '—')}")
            with action_col:
                if st.button("👁️ Inspect experiment", key="dash_inspect", width="stretch"):
                    st.session_state["compare_selected_id"] = best.get("id")
                    _go_to(st, NAV_EXPERIMENTS)
                if st.button("⚡ Use for inference", key="dash_use_inference", type="primary", width="stretch"):
                    model_key, module_key, technique_key = inference_widget_keys("infer")
                    st.session_state[model_key] = best.get("model_id", "baseline")
                    st.session_state[module_key] = best.get("module", "all")
                    st.session_state[technique_key] = best.get("technique", "all")
                    st.session_state[f"infer_technique_sync"] = best.get("technique", "all")
                    st.session_state["infer_experiment"] = best.get("id")
                    _go_to(st, NAV_IMAGE_INFERENCE)

    st.subheader("Workflow")
    workflow = [
        (NAV_EXPERIMENTS, "Compare experiments", "Evaluate and compare preprocessing experiments."),
        (NAV_IMAGE_INFERENCE, "Image processing", "Run inference on PCB images using a selected model."),
        (NAV_ANALYSIS, "Analysis & reports", "Explore results and generate performance reports."),
        (NAV_VIDEO, "Video processing", "Run inference and analysis on PCB inspection videos."),
    ]
    for page, title, caption in workflow:
        with st.container(border=True):
            text_col, button_col = st.columns([5, 1])
            text_col.markdown(f"**{title}**  \n{caption}")
            if button_col.button("→", key=f"dash_workflow_{page}"):
                _go_to(st, page)

    st.subheader("Performance overview")
    payload = build_analysis_payload(records)
    overview = pd.DataFrame(payload["original_vs_combined"])
    if not overview.empty:
        st.bar_chart(
            overview.set_index("label")[["map50_95"]].rename(columns={"map50_95": "mAP50-95"}),
            height=300,
        )


def main(results_path: Path) -> None:
    import streamlit as st

    results_path = _resolve_results_path(results_path)
    st.set_page_config(page_title="HRIPCB Preprocessing Lab", page_icon="🔬", layout="wide")
    st.markdown("""
    <style>
    .stApp { background: #f4f7fb; color: #172033; }
    .block-container { max-width: 1500px; padding-top: 2.2rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #dce5ef; border-radius: 16px; padding: 12px 16px; box-shadow: 0 12px 32px rgba(40,64,92,.07); }
    button[kind="primary"] { background: #2563eb; }
    /* File uploader's own "Browse files" button: forced blue and centered
       within its dropzone (Streamlit's own :disabled/default style otherwise
       wins and shows plain gray, left-aligned). */
    [data-testid="stFileUploaderDropzone"] { justify-content: center; }
    [data-testid="stFileUploaderDropzone"] button {
        background: #2563eb !important;
        border-color: #2563eb !important;
        color: white !important;
        margin: 0 auto;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: white !important;
    }
    [data-testid="stTabs"] button[role="tab"] { color: #172033; }
    section[data-testid="stSidebar"] button { justify-content: flex-start; text-align: left; }
    /* Nav list: buttons stacked with no gaps, no background, hover/active highlight only. */
    .st-key-nav_list [data-testid="stVerticalBlock"] { gap: 0rem; }
    .st-key-nav_list div[data-testid="stButton"] > button {
        background: transparent;
        border: none;
        border-radius: 0;
        box-shadow: none;
        color: #172033;
        font-weight: 500;
        padding: 0.6rem 0.9rem;
    }
    .st-key-nav_list div[data-testid="stButton"] > button:hover {
        background: #e7edf6;
        color: #172033;
    }
    .st-key-nav_list div[data-testid="stButton"] > button[kind="primary"] {
        background: #e2e8f5;
        color: #1d4ed8;
        border: none;
    }
    .st-key-nav_list div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #d7e1f5;
    }
    </style>
    """, unsafe_allow_html=True)

    records = _load_records(results_path)
    if not records:
        st.error(f"No experiment records were found at {results_path}.")
        return

    page = _render_sidebar_nav(st)

    if page == NAV_DASHBOARD:
        _render_dashboard_home(st, records)
    elif page == NAV_EXPERIMENTS:
        _render_comparison_mode(st, records, results_path)
    elif page == NAV_IMAGE_INFERENCE:
        _render_inference_mode(st, records)
    elif page == NAV_ANALYSIS:
        _render_analysis(st, records)
        _render_report_tools(st, records)
    elif page == NAV_VIDEO:
        _render_video_mode(st, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    args, _ = parser.parse_known_args()
    main(args.results)
