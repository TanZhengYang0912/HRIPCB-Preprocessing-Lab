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
    filter_formal_summary_rows,
    find_matching_label,
    filter_summary_rows_for_selection,
    load_ground_truth,
    load_summary_rows,
    describe_summary_condition,
    draw_detections,
    predict_image,
    prepare_condition,
    save_demo_artifacts,
)
from hripcb_baseline.member3_formal import (  # noqa: E402
    FORMAL_BILATERAL_PRESETS,
    FORMAL_GAMMAS,
    FormalCondition,
    apply_formal_condition,
    build_formal_conditions,
)


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "runs/baseline/weights/best.pt"
DEFAULT_SUMMARY = PROJECT_ROOT / "runs/member3/comparison.csv"
DEFAULT_FORMAL_SUMMARY = PROJECT_ROOT / "runs/member3_formal/comparison.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/member3_demo"
DEFAULT_SIGMAS = (10, 25, 40)
FORMAL_TECHNIQUES = {
    "Original": "original",
    "Bilateral Filtering": "bilateral",
    "AGCWD + gamma": "agcwd_gamma",
    "Bilateral Filtering + AGCWD + gamma": "combined",
}


@st.cache_resource(show_spinner="Loading YOLOv8s checkpoint...")
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _default_dataset_root() -> str:
    local_root = PROJECT_ROOT / "HRIPCB_UPDATE"
    return str(local_root) if local_root.is_dir() else ""


def _summary_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    columns = (
        ("condition", "Experiment condition"),
        ("sigma", "Noise sigma (σ)"),
        ("bilateral", "Bilateral filter"),
        ("alpha", "AGCWD alpha"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 score"),
        ("map50", "mAP@0.5"),
        ("map50_95", "mAP@0.5:0.95"),
    )
    table: list[dict[str, Any]] = []
    for row in rows:
        compact: dict[str, Any] = {}
        for source_column, display_column in columns:
            value = row.get(source_column, "")
            if source_column == "condition":
                compact[display_column] = describe_summary_condition(value)
            elif source_column in {"precision", "recall", "f1", "map50", "map50_95"}:
                try:
                    compact[display_column] = round(float(value), 4)
                except (TypeError, ValueError):
                    compact[display_column] = value
            else:
                compact[display_column] = value
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


def _formal_summary_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    columns = (
        ("model_id", "Model ID"),
        ("checkpoint", "Checkpoint"),
        ("member", "Member"),
        ("technique", "Technique"),
        ("training_preprocessing", "Training preprocessing"),
        ("evaluation_preprocessing", "Evaluation preprocessing"),
        ("dataset_split", "Dataset split"),
        ("validation_images", "Validation images"),
        ("imgsz", "Image size"),
        ("conf", "Confidence"),
        ("iou", "IoU"),
        ("bilateral_diameter", "Bilateral diameter"),
        ("bilateral_sigma_color", "Bilateral sigmaColor"),
        ("bilateral_sigma_space", "Bilateral sigmaSpace"),
        ("agcwd_alpha", "AGCWD alpha"),
        ("gamma", "Gamma"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
        ("psnr", "PSNR"),
        ("ssim", "SSIM"),
        ("processing_time_ms", "Processing time (ms)"),
    )
    numeric = {
        "conf",
        "iou",
        "bilateral_sigma_color",
        "bilateral_sigma_space",
        "agcwd_alpha",
        "gamma",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "psnr",
        "ssim",
        "processing_time_ms",
    }
    table: list[dict[str, Any]] = []
    for row in rows:
        compact: dict[str, Any] = {}
        for source_column, display_column in columns:
            value = row.get(source_column, "")
            if source_column in numeric and value not in {"", None}:
                try:
                    compact[display_column] = round(float(value), 4)
                except (TypeError, ValueError):
                    compact[display_column] = value
            else:
                compact[display_column] = value
        table.append(compact)
    return table


def _formal_condition_for_selection(
    technique: str,
    bilateral_index: int | None,
    gamma: float,
) -> FormalCondition:
    bilateral = (
        FORMAL_BILATERAL_PRESETS[bilateral_index]
        if bilateral_index is not None
        else None
    )
    for condition in build_formal_conditions():
        if (
            condition.technique == technique
            and condition.bilateral == bilateral
            and condition.gamma == gamma
        ):
            return condition
    raise ValueError("unable to resolve formal Member 3 condition")


def _run_formal_condition(
    model: YOLO,
    model_path: Path,
    original: np.ndarray,
    source_name: str,
    condition: FormalCondition,
    confidence: float,
    output_root: Path,
    label_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], float, Path]:
    processed = apply_formal_condition(original, condition)
    started = time.perf_counter()
    raw_detections = predict_image(model, processed, conf=0.05, imgsz=1024)
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
            "condition_id": condition.identifier,
            "technique": condition.technique,
            "evaluation_preprocessing": condition.description,
            "bilateral": condition.bilateral.tag if condition.bilateral else None,
            "agcwd_alpha": 0.75 if condition.technique != "original" else None,
            "gamma": condition.gamma,
            "confidence_threshold": confidence,
            "inference_confidence": 0.05,
            "imgsz": 1024,
            "model_path": str(model_path),
            "inference_ms": inference_ms,
            "detections": detections,
            "ground_truth": ground_truth,
        },
    )
    return processed, prediction, detections, inference_ms, output_dir


def _save_formal_failure_artifacts(
    original: np.ndarray,
    source_name: str,
    condition: FormalCondition,
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
            "condition_id": condition.identifier,
            "evaluation_preprocessing": condition.description,
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
    st.caption("Formal validation tuning: Bilateral Filtering + AGCWD + gamma")

    with st.sidebar:
        st.header("Formal Member 3 controls")
        uploaded_file = st.file_uploader(
            "Upload one PCB image",
            type=["jpg", "jpeg", "png"],
        )
        technique_label = st.selectbox(
            "Preprocessing technique", list(FORMAL_TECHNIQUES)
        )
        technique = FORMAL_TECHNIQUES[technique_label]
        bilateral_index: int | None = None
        if technique in {"bilateral", "combined"}:
            bilateral_index = st.selectbox(
                "Bilateral preset",
                range(len(FORMAL_BILATERAL_PRESETS)),
                format_func=lambda index: (
                    f"d={FORMAL_BILATERAL_PRESETS[index].diameter}, "
                    f"sigmaColor={FORMAL_BILATERAL_PRESETS[index].sigma_color:g}, "
                    f"sigmaSpace={FORMAL_BILATERAL_PRESETS[index].sigma_space:g}"
                ),
            )
        gamma = 1.0
        if technique in {"agcwd_gamma", "combined"}:
            gamma = st.selectbox("Gamma", FORMAL_GAMMAS)
        selected_condition = _formal_condition_for_selection(
            technique,
            bilateral_index,
            gamma,
        )
        show_all_summary = st.checkbox(
            "Show all formal experiments",
            value=False,
            help=(
                "By default, the validation table follows the selected formal "
                "condition. Turn this on to compare all 16 Member 3 experiments."
            ),
        )
        confidence = st.slider(
            "Display confidence threshold",
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
        "Upload a PCB image, choose one formal Member 3 preset, and run the frozen YOLOv8s detector."
    )

    all_summary_rows = load_summary_rows(DEFAULT_FORMAL_SUMMARY, split=None)
    formal_validation_rows = [
        row for row in all_summary_rows if row.get("dataset_split") == "val"
    ]
    summary_rows = (
        formal_validation_rows
        if show_all_summary
        else filter_formal_summary_rows(
            formal_validation_rows,
            selected_condition.identifier,
        )
    )
    summary_scope = (
        "all 16 formal experiments"
        if show_all_summary
        else f"selected setup: {selected_condition.description}"
    )
    with st.expander(
        f"Experiment Summary (validation tuning) — {summary_scope}",
        expanded=True,
    ):
        st.info(
            (
                "Showing all 16 precomputed formal Member 3 validation experiments."
                if show_all_summary
                else (
                    "This table follows the selected formal preset and shows its "
                    "dataset-level validation result. It is separate from the "
                    "single-image display below."
                )
            )
        )
        st.caption(
            "All formal rows use the frozen baseline checkpoint, image size 1024, "
            "confidence 0.25, IoU 0.70, and validation split only."
        )
        if summary_rows:
            st.dataframe(
                _formal_summary_table(summary_rows),
                use_container_width=True,
            )
        else:
            st.info(
                "No formal validation result is available yet. Run "
                "scripts/run_member3_formal.py after the full dataset is available."
            )

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
            processed, prediction, detections, inference_ms, output_dir = _run_formal_condition(
                model,
                model_path,
                original,
                uploaded_file.name,
                selected_condition,
                float(confidence),
                DEFAULT_OUTPUT,
                label_path,
            )
            columns = st.columns(3)
            columns[0].image(
                original,
                caption="Original input image",
                use_container_width=True,
            )
            columns[1].image(
                processed,
                caption=f"Processed image — {selected_condition.description}",
                use_container_width=True,
            )
            columns[2].image(
                prediction,
                caption=(
                    "YOLOv8s detection result — "
                    f"{selected_condition.description}"
                ),
                use_container_width=True,
            )

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
            metric_columns[2].metric("Condition", selected_condition.description)
            if detections:
                st.dataframe(detections, use_container_width=True)
            else:
                st.info("No defects were detected above the selected confidence threshold.")
            st.success(f"Saved artefacts to {output_dir}")
        except Exception as exc:  # pragma: no cover - exercised by model runtime
            failure_dir = _save_formal_failure_artifacts(
                original,
                uploaded_file.name,
                selected_condition,
                exc,
            )
            st.error(f"Detection failed: {exc}")
            st.caption(f"Uploaded image preserved at {failure_dir}")

    if compare_all:
        st.subheader("Compare All Formal Conditions")
        for comparison_condition in build_formal_conditions():
            try:
                _, prediction, detections, inference_ms, output_dir = _run_formal_condition(
                    model,
                    model_path,
                    original,
                    uploaded_file.name,
                    comparison_condition,
                    float(confidence),
                    DEFAULT_OUTPUT,
                    label_path,
                )
                st.image(
                    prediction,
                    caption=(
                        f"{comparison_condition.description} | "
                        f"{len(detections)} detections | "
                        f"{inference_ms:.1f} ms"
                    ),
                    use_container_width=True,
                )
                st.caption(f"Saved to {output_dir}")
            except Exception as exc:  # pragma: no cover - exercised by model runtime
                failure_dir = _save_formal_failure_artifacts(
                    original,
                    uploaded_file.name,
                    comparison_condition,
                    exc,
                )
                st.error(f"{comparison_condition.description} failed: {exc}")
                st.caption(f"Uploaded image preserved at {failure_dir}")


if __name__ == "__main__":
    main()
