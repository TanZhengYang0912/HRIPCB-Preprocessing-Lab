#!/usr/bin/env python3
"""Interactive Streamlit view over the same generic project results JSON."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hripcb_member1.evaluation import select_device
from hripcb_dashboard.batch import extract_image_entries
from hripcb_dashboard.reporting import build_report_pdf, dumps_json, record_metric_summary
from hripcb_dashboard.video import process_video
from hripcb_preprocessing.candidates import apply_candidate
from hripcb_dashboard.filtering import (
    FILTER_FIELDS,
    best_by_module,
    best_experiment,
    filter_records,
    inference_widget_keys,
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


def _load_records(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _label(key: str) -> str:
    return METRIC_LABELS.get(key, key.replace("_", " ").title())


def _checkpoint_for_model(model_id: str) -> Path:
    if model_id == "final":
        return PROJECT_ROOT / "runs/final_model/weights/best.pt"
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
    st.header("Analysis & findings")
    st.caption("A report-ready overview of coverage, ranking, and measurable model differences.")
    summary = record_metric_summary(records)
    cards = st.columns(4)
    cards[0].metric("Experiment runs", summary["count"])
    cards[1].metric("Modules", summary["module_count"])
    cards[2].metric("Models", summary["model_count"])
    best = summary["best"]
    cards[3].metric("Highest mAP50-95", f"{_metric_value(best, 'map50_95'):.4f}" if best else "—")
    st.caption(
        f"Coverage: {summary['module_count']} member modules + "
        f"{summary['baseline_control_count']} baseline control."
    )
    if best:
        st.success(
            f"Highest recorded run: {best.get('id', '—')} · "
            f"{best.get('module', '—')} / {best.get('technique', '—')}"
        )
    rows = []
    for record in summary["ranked"]:
        rows.append({
            "Rank": len(rows) + 1,
            "ID": record.get("id", "—"),
            "Model": record.get("model_id", "baseline"),
            "Module": record.get("module", "—"),
            "Technique": record.get("technique", "—"),
            "Split": record.get("split", "—"),
            "mAP50-95": round(_metric_value(record, "map50_95"), 4),
            "mAP50": round(_metric_value(record, "map50"), 4),
            "F1": round(_metric_value(record, "f1"), 4),
            "Precision": round(_metric_value(record, "precision"), 4),
            "Recall": round(_metric_value(record, "recall"), 4),
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    baseline = next((record for record in records if record.get("id") == "baseline_original_test"), None)
    final = next((record for record in records if record.get("model_id") == "final" and record.get("evaluation_type") == "official_final"), None)
    if baseline and final:
        st.subheader("Final model vs baseline")
        comparison = []
        for key in ("precision", "recall", "map50", "map50_95", "f1"):
            baseline_value = _metric_value(baseline, key)
            final_value = _metric_value(final, key)
            comparison.append({
                "Metric": _label(key),
                "Baseline": round(baseline_value, 4),
                "Final": round(final_value, 4),
                "Difference": round(final_value - baseline_value, 4),
            })
        st.dataframe(comparison, width="stretch", hide_index=True)


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


def _detect(model, image: np.ndarray) -> tuple[np.ndarray, int]:
    result = model.predict(
        source=image,
        imgsz=INFERENCE_IMGSZ,
        conf=INFERENCE_CONF,
        iou=INFERENCE_IOU,
        device=select_device("auto"),
        verbose=False,
    )[0]
    plotted = result.plot()
    count = int(len(result.boxes)) if result.boxes is not None else 0
    return cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), count


def _option_label(value: str) -> str:
    return "All" if value == "all" else value


def _select_value(st, label: str, values: list[str], *, key: str) -> str:
    choices = ["all", *values]
    current = st.session_state.get(key, "all")
    if current not in choices:
        current = "all"
        st.session_state[key] = current
    kwargs = {"format_func": _option_label, "key": key}
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
            st, "Module", module_options, key=keys["module"]
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

    model_col, module_col, technique_col = st.columns(3)
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
    with technique_col:
        technique_options = option_values(
            records, model=selection["model"], module=selection["module"]
        )["technique"]
        selection["technique"] = _select_value(
            st, "Technique", technique_options, key=technique_key
        )
    selection = normalize_selection(records, selection)
    return selection


def _render_active_experiment(st, record: dict, *, heading: str = "Active experiment") -> None:
    st.subheader(heading)
    st.caption(
        f"{record.get('model_label', record.get('model_id', 'baseline'))} · "
        f"{record.get('module', '—')} · {record.get('technique', '—')} · {record.get('id', '—')}"
    )
    cards = st.columns(5)
    cards[0].metric("Image size", str(INFERENCE_IMGSZ))
    cards[1].metric("Confidence", f"{INFERENCE_CONF:.2f}")
    cards[2].metric("IoU", f"{INFERENCE_IOU:.2f}")
    cards[3].metric("Split", str(record.get("split", "—")))
    cards[4].metric("mAP50-95", f"{float(record.get('metrics', {}).get('map50_95', 0)):.4f}")
    with st.expander("Exact preprocessing parameters", expanded=True):
        st.json(record.get("parameters", {}))


def _render_recommendation(st, records: list[dict], *, key_prefix: str = "infer") -> None:
    recommended = best_experiment(records)
    st.subheader("Recommended best experiment")
    st.caption("Recommendation rule: val split + ablation evaluation + highest mAP50-95.")
    if recommended is None:
        st.warning("No validation ablation result is available for recommendation.")
        return

    score = float(recommended.get("metrics", {}).get("map50_95", 0))
    st.success(
        f"{recommended.get('id', '—')} · "
        f"{recommended.get('model_label', recommended.get('model_id', 'baseline'))} · "
        f"{recommended.get('module', '—')} / {recommended.get('technique', '—')}"
    )
    cards = st.columns(4)
    cards[0].metric("Recommended mAP50-95", f"{score:.4f}")
    cards[1].metric("Model", recommended.get("model_id", "baseline"))
    cards[2].metric("Module", recommended.get("module", "—"))
    cards[3].metric("Technique", recommended.get("technique", "—"))
    st.caption(f"Parameters: {json.dumps(recommended.get('parameters', {}), sort_keys=True)}")
    if st.button("Use recommended experiment", key=f"use_recommended_{key_prefix}"):
        model_key, module_key, technique_key = inference_widget_keys(key_prefix)
        st.session_state[model_key] = recommended.get("model_id", "baseline")
        st.session_state[module_key] = recommended.get("module", "all")
        st.session_state[technique_key] = recommended.get("technique", "all")
        st.session_state[f"{key_prefix}_experiment"] = recommended.get("id")
        st.rerun()

    module_rows = best_by_module(records)
    if module_rows:
        with st.expander("Best experiment by module"):
            st.dataframe(
                [
                    {
                        "Module": row.get("module", "—"),
                        "Technique": row.get("technique", "—"),
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
    st.caption("Filter the shared validation and test records. Technique options follow the selected module.")
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
            kwargs={"state": st.session_state, "prefix": "compare_"},
        )

    filtered = filter_records(records, **selection)
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

    best = filtered[0]
    cards = st.columns(4)
    cards[0].metric("Visible runs", len(filtered))
    cards[1].metric(_label(sort_metric), f"{float(best.get('metrics', {}).get(sort_metric, 0)):.4f}")
    cards[2].metric("Best run", best.get("id", "—"))
    cards[3].metric("Module", best.get("module", "—"))

    table = []
    for record in filtered:
        row = {
            "ID": record["id"],
            "Model": record.get("model_id", "baseline"),
            "Module": record.get("module", "—"),
            "Technique": record.get("technique", "—"),
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
    preview_path = (results_path.parent / selected.get("preview", "")).resolve()
    with left:
        st.subheader(selected["id"])
        if preview_path.is_file():
            st.image(str(preview_path), caption=f"{selected.get('module')} / {selected.get('technique')}", width="stretch")
        else:
            st.info(f"Preview not found: {preview_path}")
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


def _render_inference_mode(st, records: list[dict]) -> None:
    st.header("Run image inference")
    st.caption("Select a model and preprocessing preset independently from the comparison table.")
    _render_recommendation(st, records)
    selection = _render_inference_filters(st, records)
    candidates = filter_records(
        records,
        model=selection["model"],
        module=selection["module"],
        technique=selection["technique"],
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
    selected_id = st.selectbox("Parameter preset / experiment", experiment_ids, **experiment_kwargs)
    selected = next(record for record in candidates if record["id"] == selected_id)
    _render_active_experiment(st, selected)
    selected_model = selected.get("model_id", "baseline")
    _render_reproducibility(st, selected, selected_model, key_prefix="infer")
    selected_checkpoint = _checkpoint_for_model(selected_model)
    if not selected_checkpoint.is_file():
        st.error(f"Checkpoint not available for {selected_model}: {selected_checkpoint}")
        return
    uploads = st.file_uploader(
        "Upload PCB images or one ZIP folder",
        type=["jpg", "jpeg", "png", "zip"],
        accept_multiple_files=True,
        key="image_inference_uploads",
    )
    if uploads and st.button("Run detection on uploaded images", key="run_inference"):
        image_entries, skipped = extract_image_entries(
            [(upload.name, upload.getvalue()) for upload in uploads]
        )
        if skipped:
            st.warning("Some files were skipped:\n\n" + "\n".join(f"- {message}" for message in skipped))
        st.info(f"Prepared {len(image_entries)} image(s) for detection; skipped {len(skipped)} file(s).")
        model = _load_model(str(selected_checkpoint))
        summary = []
        visual_results = []
        for filename, payload in image_entries:
            try:
                original = _decode_payload(filename, payload)
                processed = apply_candidate(original, _candidate_from_record(selected))
                plotted, count = _detect(model, processed)
            except (ValueError, cv2.error) as error:
                st.error(f"{filename}: {error}")
                continue
            summary.append({
                "file": filename,
                "detections": count,
                "model": selected_model,
                "experiment": selected["id"],
            })
            visual_results.append({
                "file": filename,
                "detections": count,
                "original": cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                "processed": cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                "result": plotted,
            })
        if summary:
            st.dataframe(summary, width="stretch", hide_index=True)
            st.download_button(
                "Download inference summary",
                dumps_json(summary),
                file_name="inference_summary.json",
                mime="application/json",
                key="inference_download",
            )
        st.subheader("Visual results for every uploaded image")
        for index, item in enumerate(visual_results):
            with st.expander(f"{item['file']} - {item['detections']} detections", expanded=index == 0):
                c1, c2, c3 = st.columns(3)
                c1.image(item["original"], caption="Uploaded original", width="stretch")
                c2.image(item["processed"], caption="After selected preprocessing", width="stretch")
                c3.image(item["result"], caption="YOLO detection result", width="stretch")


def _render_video_mode(st, records: list[dict]) -> None:
    st.header("Video processing")
    st.caption("Run the selected preprocessing and YOLO model frame by frame, then download the annotated video.")
    _render_recommendation(st, records, key_prefix="video")
    selection = _render_inference_filters(st, records, key_prefix="video")
    candidates = filter_records(
        records,
        model=selection["model"],
        module=selection["module"],
        technique=selection["technique"],
    )
    if not candidates:
        st.warning("No video preset matches the selected model, module, and technique.")
        return
    experiment_ids = [record["id"] for record in candidates]
    current_id = st.session_state.get("video_experiment", experiment_ids[0])
    if current_id not in experiment_ids:
        current_id = experiment_ids[0]
    selected_id = st.selectbox("Video parameter preset / experiment", experiment_ids, index=experiment_ids.index(current_id), key="video_experiment")
    selected = next(record for record in candidates if record["id"] == selected_id)
    _render_active_experiment(st, selected, heading="Selected video experiment")
    selected_model = selected.get("model_id", "baseline")
    _render_reproducibility(st, selected, selected_model, key_prefix="video")
    checkpoint = _checkpoint_for_model(selected_model)
    if not checkpoint.is_file():
        st.error(f"Checkpoint not available for {selected_model}: {checkpoint}")
        return
    video = st.file_uploader(
        "Upload a short video",
        type=["mp4", "mov", "avi"],
        accept_multiple_files=False,
        key="video_inference_upload",
    )
    if video and st.button("Run video detection", key="run_video_detection"):
        model = _load_model(str(checkpoint))
        progress = st.progress(0, text="Preparing video...")
        try:
            with tempfile.TemporaryDirectory(prefix="hripcb_video_") as temp_dir:
                input_path = Path(temp_dir) / video.name
                output_path = Path(temp_dir) / "annotated_output.mp4"
                input_path.write_bytes(video.getvalue())

                def update_progress(done: int, total: int) -> None:
                    if total > 0:
                        progress.progress(min(done / total, 1.0), text=f"Processing frame {done}/{total}")

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
                output_bytes = output_path.read_bytes()
            progress.progress(1.0, text="Video processing complete")
        except (ValueError, cv2.error) as error:
            progress.empty()
            st.error(str(error))
            return
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


def main(results_path: Path) -> None:
    import streamlit as st

    st.set_page_config(page_title="HRIPCB Preprocessing Lab", page_icon="🔬", layout="wide")
    st.markdown("""
    <style>
    .stApp { background: #f4f7fb; color: #172033; }
    .block-container { max-width: 1500px; padding-top: 2.2rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #dce5ef; border-radius: 16px; padding: 12px 16px; box-shadow: 0 12px 32px rgba(40,64,92,.07); }
    button[kind="primary"] { background: #2563eb; }
    [data-testid="stTabs"] button[role="tab"] { color: #172033; }
    </style>
    """, unsafe_allow_html=True)

    records = _load_records(results_path)
    st.title("HRIPCB Preprocessing Lab")
    st.caption("A shared, report-ready workspace for Member 1–4 preprocessing experiments.")
    if not records:
        st.error(f"No experiment records were found at {results_path}.")
        return
    tabs = st.tabs(["Compare experiments", "Run image inference", "Analysis & reports", "Video processing"])
    with tabs[0]:
        _render_comparison_mode(st, records, results_path)
    with tabs[1]:
        _render_inference_mode(st, records)
    with tabs[2]:
        _render_analysis(st, records)
        _render_report_tools(st, records)
    with tabs[3]:
        _render_video_mode(st, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", type=Path, default=Path("runs/project_validation_comparison/results.json"))
    args, _ = parser.parse_known_args()
    main(args.results)
