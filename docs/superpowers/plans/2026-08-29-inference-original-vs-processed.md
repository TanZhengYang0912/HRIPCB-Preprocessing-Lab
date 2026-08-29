# Original vs Processed Detection Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show YOLO detection results for both the original and the preprocessed image side by side in the Streamlit inference tab, so the effect of preprocessing on detection is directly visible.

**Architecture:** Run `_detect` twice per uploaded image, once on the decoded original and once on the preprocessed variant. Extract the summary-row construction into a pure, testable helper in `hripcb_dashboard.analysis`, then render a 2x2 panel grid pairing each image with its own detection output.

**Tech Stack:** Python 3.12, Streamlit, Ultralytics YOLO, OpenCV, pytest.

**Spec:** No separate spec document. This was scoped as a bounded change during brainstorming; the agreed design is: run detection on both images, add original/processed/change columns to the summary table, and render a 2x2 layout.

## Global Constraints

- Do not remove any string asserted by `tests/test_dashboard_extra_effort.py::test_streamlit_exposes_extra_effort_sections_and_frozen_protocol`.
- `scripts/streamlit_dashboard.py` must remain importable without a Streamlit runtime, since `tests/test_streamlit_deployment.py` imports it directly.
- Run tests with `PYTHONPATH=src ./.venv/bin/python -m pytest`.
- Preprocessing must continue to be applied via `apply_candidate(original, _candidate_from_record(selected))`; do not inline filter calls.
- The JSON download payload must stay serialisable, so the summary rows must contain only plain str and int values.

---

## File Structure

- `src/hripcb_dashboard/analysis.py` — gains `detection_comparison_row`, a pure function building one summary row. Lives here because this module already owns result-presentation helpers (`technique_label`, `metric_label`).
- `scripts/streamlit_dashboard.py` — `_render_inference_mode` runs detection twice and renders the 2x2 grid. Remains a thin UI shell with no new logic of its own.
- `tests/test_dashboard_inference_comparison.py` — new test file covering the helper and the rendering wiring.

---

### Task 1: Summary row helper

**Files:**
- Modify: `src/hripcb_dashboard/analysis.py` (append at end of file)
- Test: `tests/test_dashboard_inference_comparison.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `detection_comparison_row(file_name: str, original_count: int, processed_count: int, model_id: str, experiment_id: str) -> dict[str, str | int]` returning keys `file`, `original`, `processed`, `change`, `model`, `experiment`. Task 2 imports this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dashboard_inference_comparison.py`:

```python
from hripcb_dashboard.analysis import detection_comparison_row


def test_row_reports_both_counts_and_signed_change():
    row = detection_comparison_row("board.jpg", 5, 7, "baseline", "wavelet_w_sym4")
    assert row == {
        "file": "board.jpg",
        "original": 5,
        "processed": 7,
        "change": "+2",
        "model": "baseline",
        "experiment": "wavelet_w_sym4",
    }


def test_row_marks_a_drop_with_a_minus_sign():
    row = detection_comparison_row("board.jpg", 6, 4, "baseline", "homomorphic_h_c30")
    assert row["change"] == "-2"


def test_row_reports_no_change_without_a_sign():
    row = detection_comparison_row("board.jpg", 3, 3, "baseline", "original")
    assert row["change"] == "0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_dashboard_inference_comparison.py -v`
Expected: FAIL with `ImportError: cannot import name 'detection_comparison_row'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/hripcb_dashboard/analysis.py`:

```python
def detection_comparison_row(
    file_name: str,
    original_count: int,
    processed_count: int,
    model_id: str,
    experiment_id: str,
) -> dict[str, str | int]:
    """Build one summary row comparing detections before and after preprocessing."""

    original_count = int(original_count)
    processed_count = int(processed_count)
    difference = processed_count - original_count
    change = "0" if difference == 0 else f"{difference:+d}"
    return {
        "file": str(file_name),
        "original": original_count,
        "processed": processed_count,
        "change": change,
        "model": str(model_id),
        "experiment": str(experiment_id),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_dashboard_inference_comparison.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add src/hripcb_dashboard/analysis.py tests/test_dashboard_inference_comparison.py
git commit -m "feat: add detection comparison summary row helper"
```

---

### Task 2: Detect on both images and render the 2x2 grid

**Files:**
- Modify: `scripts/streamlit_dashboard.py:684-717` (inside `_render_inference_mode`)
- Modify: `scripts/streamlit_dashboard.py` import block (add `detection_comparison_row`)
- Test: `tests/test_dashboard_inference_comparison.py` (append)

**Interfaces:**
- Consumes: `detection_comparison_row` from Task 1.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard_inference_comparison.py`:

```python
from pathlib import Path


def test_inference_mode_detects_on_original_and_processed():
    source = Path("scripts/streamlit_dashboard.py").read_text(encoding="utf-8")

    for fragment in (
        "detection_comparison_row",
        "original_plotted, original_count = _detect(model, original)",
        "processed_plotted, processed_count = _detect(model, processed)",
        "Detection on original",
        "Detection after preprocessing",
    ):
        assert fragment in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_dashboard_inference_comparison.py::test_inference_mode_detects_on_original_and_processed -v`
Expected: FAIL on the first missing fragment

- [ ] **Step 3: Add the import**

In `scripts/streamlit_dashboard.py`, find the existing line:

```python
from hripcb_dashboard.analysis import MEMBER_TECHNIQUES, build_analysis_payload, technique_label
```

Replace it with:

```python
from hripcb_dashboard.analysis import (
    MEMBER_TECHNIQUES,
    build_analysis_payload,
    detection_comparison_row,
    technique_label,
)
```

- [ ] **Step 4: Replace the detection loop**

In `scripts/streamlit_dashboard.py`, replace lines 682 to 702 (the `for filename, payload in image_entries:` body through the end of `visual_results.append({...})`) with:

```python
        for filename, payload in image_entries:
            try:
                original = _decode_payload(filename, payload)
                original_plotted, original_count = _detect(model, original)
                processed = apply_candidate(original, _candidate_from_record(selected))
                processed_plotted, processed_count = _detect(model, processed)
            except (ValueError, cv2.error) as error:
                st.error(f"{filename}: {error}")
                continue
            summary.append(
                detection_comparison_row(
                    filename,
                    original_count,
                    processed_count,
                    selected_model,
                    selected["id"],
                )
            )
            visual_results.append({
                "file": filename,
                "original_count": original_count,
                "processed_count": processed_count,
                "original": cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                "processed": cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                "original_result": original_plotted,
                "processed_result": processed_plotted,
            })
```

- [ ] **Step 5: Replace the visual panel block**

In `scripts/streamlit_dashboard.py`, replace the `for index, item in enumerate(visual_results):` block (the expander with `c1, c2, c3`) with:

```python
        for index, item in enumerate(visual_results):
            header = (
                f"{item['file']} - original {item['original_count']} detections, "
                f"after preprocessing {item['processed_count']} detections"
            )
            with st.expander(header, expanded=index == 0):
                top_left, top_right = st.columns(2)
                top_left.image(item["original"], caption="Uploaded original", width="stretch")
                top_right.image(item["original_result"], caption="Detection on original", width="stretch")
                bottom_left, bottom_right = st.columns(2)
                bottom_left.image(item["processed"], caption="After selected preprocessing", width="stretch")
                bottom_right.image(
                    item["processed_result"],
                    caption="Detection after preprocessing",
                    width="stretch",
                )
```

- [ ] **Step 6: Run the new test to verify it passes**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_dashboard_inference_comparison.py -v`
Expected: PASS, 4 tests

- [ ] **Step 7: Run the full suite to confirm nothing regressed**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest -q`
Expected: PASS, 82 tests (78 existing plus 4 new)

- [ ] **Step 8: Confirm the module still imports without Streamlit running**

Run: `PYTHONPATH=src ./.venv/bin/python -c "import scripts.streamlit_dashboard as d; print('import OK')"`
Expected: `import OK`

- [ ] **Step 9: Commit**

```bash
git add scripts/streamlit_dashboard.py tests/test_dashboard_inference_comparison.py
git commit -m "feat: show detection on original alongside preprocessed image"
```

---

## Manual verification

After Task 2, start the app and check the inference tab by hand, since the panel layout cannot be asserted from source alone:

```bash
PYTHONPATH=src ./.venv/bin/streamlit run scripts/streamlit_dashboard.py -- --results runs/project_validation_comparison/results.json
```

Confirm all four of these:

- Upload one PCB image, pick module `member2` and technique `wavelet_homomorphic`, then press Run detection.
- The expander header shows two counts, one for original and one for after preprocessing.
- Four panels appear in a 2x2 grid, with detections drawn on both right-hand panels.
- The summary table has `original`, `processed` and `change` columns, and the JSON download still parses.

## Known cost

Detection now runs twice per uploaded image, so inference time roughly doubles. This is inherent to showing both results and is not a defect.
