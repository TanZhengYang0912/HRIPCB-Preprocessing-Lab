#!/usr/bin/env python3
"""Launch the local Streamlit dashboard for interactive Member 3 testing."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hripcb_baseline.member3_demo import (  # noqa: E402
    CONDITION_LABELS,
    DEMO_AGCWD_ALPHA,
    DEMO_BILATERAL,
    filter_detections,
    find_matching_label,
    load_ground_truth,
    load_summary_rows,
    draw_detections,
    predict_image,
    prepare_condition,
    save_demo_artifacts,
)


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "runs/baseline/weights/best.pt"
DEFAULT_SUMMARY = PROJECT_ROOT / "runs/member3/comparison.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/member3_demo"
DEFAULT_SIGMAS = (10, 25, 40)


@st.cache_resource(show_spinner="Loading YOLOv8s checkpoint...")
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _default_dataset_root() -> str:
    local_root = PROJECT_ROOT / "HRIPCB_UPDATE"
    return str(local_root) if local_root.is_dir() else ""


def _summary_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    columns = (
        "condition",
        "sigma",
        "bilateral",
        "alpha",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
    )
    table: list[dict[str, Any]] = []
    for row in rows:
        compact: dict[str, Any] = {}
        for column in columns:
            value = row.get(column, "")
            if column in {"precision", "recall", "f1", "map50", "map50_95"}:
                try:
                    compact[column] = round(float(value), 4)
                except (TypeError, ValueError):
                    compact[column] = value
            else:
                compact[column] = value
        table.append(compact)
    return table


def _run_single_condition(
    model: YOLO,
    model_path: Path,
    original: np.ndarray,
    source_name: str,
    condition: str,
    sigma: int,
    confidence: float,
    output_root: Path,
    label_path: Path | None,
) -> tuple[np.ndarray, list[dict[str, object]], float, Path]:
    processed = prepare_condition(original, condition, sigma=sigma)
    started = time.perf_counter()
    raw_detections = predict_image(
        model,
        processed,
        conf=0.05,
        imgsz=1024,
    )
    inference_ms = (time.perf_counter() - started) * 1000.0
    detections = filter_detections(raw_detections, confidence)
    prediction = draw_detections(processed, detections)
    ground_truth = (
        load_ground_truth(label_path, original.shape)
        if label_path is not None
        else []
    )
    output_dir = save_demo_artifacts(
        output_root,
        original=original,
        processed=processed,
        prediction=prediction,
        source_name=source_name,
        metadata={
            "condition": condition,
            "sigma": sigma,
            "bilateral": DEMO_BILATERAL.tag,
            "alpha": DEMO_AGCWD_ALPHA,
            "confidence_threshold": confidence,
            "inference_confidence": 0.05,
            "imgsz": 1024,
            "model_path": str(model_path),
            "inference_ms": inference_ms,
            "detections": detections,
            "ground_truth": ground_truth,
        },
    )
    return prediction, detections, inference_ms, output_dir


def _save_failure_artifacts(
    original: np.ndarray,
    source_name: str,
    condition: str,
    sigma: int,
    error: Exception,
) -> Path:
    return save_demo_artifacts(
        DEFAULT_OUTPUT,
        original=original,
        processed=original,
        prediction=original,
        source_name=source_name,
        metadata={
            "status": "error",
            "condition": condition,
            "sigma": sigma,
            "model_path": str(DEFAULT_MODEL),
            "error": str(error),
        },
    )


def main() -> None:
    st.set_page_config(
        page_title="Member 3 AI PCB Defect Inspector",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Member 3 AI PCB Defect Inspector")
    st.caption("Bilateral Filtering + AGCWD with a frozen YOLOv8s detector")

    with st.sidebar:
        st.header("Test Controls")
        uploaded_file = st.file_uploader(
            "Upload one PCB image",
            type=["jpg", "jpeg", "png"],
        )
        condition = st.selectbox("Preprocessing condition", CONDITION_LABELS)
        sigma = st.selectbox("Gaussian noise sigma", DEFAULT_SIGMAS, index=0)
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.90,
            value=0.25,
            step=0.05,
        )
        st.caption(f"Fixed model: `{DEFAULT_MODEL}`")
        dataset_root_text = st.text_input(
            "Dataset root (optional)",
            _default_dataset_root(),
            help="Set this to HRIPCB_UPDATE to show Ground Truth for known images.",
        )
        run_detection = st.button("Run Detection", type="primary")
        compare_all = st.button("Compare All Conditions")

    st.info(
        "Upload a PCB image, choose a Member 3 condition, and run the fixed YOLOv8s detector."
    )

    summary_rows = load_summary_rows(DEFAULT_SUMMARY, split="test")
    with st.expander("Experiment Summary (test split)", expanded=False):
        if summary_rows:
            st.dataframe(_summary_table(summary_rows), use_container_width=True)
        else:
            st.info("No comparison CSV found. Run the batch Member 3 experiment first.")

    if uploaded_file is None:
        st.warning("Upload a JPG, JPEG, or PNG image to begin.")
        return

    try:
        original = np.asarray(Image.open(uploaded_file).convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # pragma: no cover - exercised by Streamlit upload runtime
        st.error(f"Unable to read the uploaded image: {exc}")
        return

    model_path = DEFAULT_MODEL
    if not model_path.is_file():
        st.error(f"Model checkpoint not found: {model_path}")
        return

    try:
        model = load_model(str(model_path))
    except Exception as exc:  # pragma: no cover - exercised by model runtime
        st.error(f"Unable to load the YOLO checkpoint: {exc}")
        return

    dataset_root = Path(dataset_root_text).expanduser() if dataset_root_text else None
    label_path = (
        find_matching_label(dataset_root, uploaded_file.name)
        if dataset_root is not None and dataset_root.is_dir()
        else None
    )

    if run_detection:
        try:
            prediction, detections, inference_ms, output_dir = _run_single_condition(
                model,
                model_path,
                original,
                uploaded_file.name,
                condition,
                int(sigma),
                float(confidence),
                DEFAULT_OUTPUT,
                label_path,
            )
            processed = prepare_condition(original, condition, sigma=int(sigma))
            columns = st.columns(3)
            columns[0].image(original, caption="Original Image", use_container_width=True)
            columns[1].image(processed, caption="Processed Image", use_container_width=True)
            columns[2].image(prediction, caption="Detection Result", use_container_width=True)

            if label_path is not None:
                ground_truth = load_ground_truth(label_path, original.shape)
                ground_truth_image = draw_detections(
                    processed,
                    ground_truth,
                    color=(255, 0, 0),
                )
                st.image(ground_truth_image, caption="Ground Truth", use_container_width=True)

            st.subheader("Detection Details")
            metric_columns = st.columns(3)
            metric_columns[0].metric("Detections", len(detections))
            metric_columns[1].metric("Inference", f"{inference_ms:.1f} ms")
            metric_columns[2].metric("Condition", condition)
            if detections:
                st.dataframe(detections, use_container_width=True)
            else:
                st.info("No defects were detected above the selected confidence threshold.")
            st.success(f"Saved artefacts to {output_dir}")
        except Exception as exc:  # pragma: no cover - exercised by model runtime
            failure_dir = _save_failure_artifacts(
                original,
                uploaded_file.name,
                condition,
                int(sigma),
                exc,
            )
            st.error(f"Detection failed: {exc}")
            st.caption(f"Uploaded image preserved at {failure_dir}")

    if compare_all:
        st.subheader("Compare All Conditions")
        for selected_condition in CONDITION_LABELS:
            try:
                prediction, detections, inference_ms, output_dir = _run_single_condition(
                    model,
                    model_path,
                    original,
                    uploaded_file.name,
                    selected_condition,
                    int(sigma),
                    float(confidence),
                    DEFAULT_OUTPUT,
                    label_path,
                )
                st.image(
                    prediction,
                    caption=(
                        f"{selected_condition} | {len(detections)} detections | "
                        f"{inference_ms:.1f} ms"
                    ),
                    use_container_width=True,
                )
                st.caption(f"Saved to {output_dir}")
            except Exception as exc:  # pragma: no cover - exercised by model runtime
                failure_dir = _save_failure_artifacts(
                    original,
                    uploaded_file.name,
                    selected_condition,
                    int(sigma),
                    exc,
                )
                st.error(f"{selected_condition} failed: {exc}")
                st.caption(f"Uploaded image preserved at {failure_dir}")


if __name__ == "__main__":
    main()
