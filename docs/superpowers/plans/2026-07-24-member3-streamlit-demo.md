# Member 3 Streamlit Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit dashboard that applies the frozen Member 3 preprocessing pipeline to one uploaded PCB image and displays YOLOv8s detections, optional ground truth, saved artefacts, and the existing experiment summary.

**Architecture:** Keep all image-processing, inference-adapter, annotation, label-loading, CSV-loading, and persistence behavior in a UI-independent module. Keep `scripts/member3_demo.py` responsible only for Streamlit controls and rendering. Reuse `src/hripcb_baseline/member3.py` for the actual Y-channel algorithms and `runs/baseline/weights/best.pt` as the unchanged detector.

**Tech Stack:** Python 3.9+, NumPy, OpenCV, Pillow, Ultralytics YOLOv8s, Streamlit, pytest, standard-library `csv`/`json`/`pathlib`.

## Global Constraints

- Do not retrain or modify `runs/baseline/weights/best.pt`.
- Use Bilateral parameters `d=7`, `sigmaColor=75`, `sigmaSpace=75`.
- Use AGCWD `alpha=0.5`.
- Apply preprocessing to the Y channel in YCrCb and preserve Cr/Cb.
- Use noise sigmas `10`, `25`, `40` with the existing deterministic seeds.
- Use YOLO `imgsz=1024`; default confidence threshold is `0.25`.
- Keep defect class names exactly: `Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spurious_copper`, `Spur`.
- Save interactive demo artefacts below `runs/member3_demo/` and never add dataset images or generated demo output to Git.
- The official batch metrics remain in `runs/member3/comparison.csv`; single-image inference must not be presented as dataset-level mAP.

---

### Task 1: Add the UI-independent demo seams

**Files:**
- Create: `src/hripcb_baseline/member3_demo.py`
- Test: `tests/test_member3_demo.py`

**Interfaces:**
- Consumes: `src/hripcb_baseline/member3.py` preprocessing functions and the fixed class order.
- Produces: `CONDITION_LABELS`, `CLASS_NAMES`, `prepare_condition`, `predict_image`, `draw_detections`, `load_ground_truth`, `load_summary_rows`, `save_demo_artifacts` for the UI and later tests.

- [ ] **Step 1: Write the failing tests for condition dispatch and determinism.**

Add tests that use a small non-constant RGB array and assert that `prepare_condition` returns an RGB array with the same shape and `uint8` dtype for all five condition labels. Assert that two `Noisy` calls with the same sigma return equal arrays, and that `Clean` returns an equal copy of the input. Assert the invalid condition and invalid sigma errors are `ValueError`.

```python
def test_prepare_condition_supports_all_conditions():
    image = np.full((8, 10, 3), 128, dtype=np.uint8)
    image[2:6, 3:7] = [220, 80, 40]
    for condition in CONDITION_LABELS:
        output = prepare_condition(image, condition, sigma=10)
        assert output.shape == image.shape
        assert output.dtype == np.uint8

def test_prepare_condition_noise_is_reproducible():
    image = np.full((8, 10, 3), 128, dtype=np.uint8)
    first = prepare_condition(image, "Noisy", sigma=25)
    second = prepare_condition(image, "Noisy", sigma=25)
    np.testing.assert_array_equal(first, second)
```

- [ ] **Step 2: Run the focused tests and verify they fail because the module is absent.**

Run:

```bash
python3 -m pytest tests/test_member3_demo.py -q
```

Expected: collection failure with `ModuleNotFoundError: No module named 'hripcb_baseline.member3_demo'`.

- [ ] **Step 3: Implement the condition dispatcher.**

Define:

```python
CONDITION_LABELS = ("Clean", "Noisy", "Bilateral Filtering", "AGCWD", "Bilateral + AGCWD")
CLASS_NAMES = ("Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spurious_copper", "Spur")

def prepare_condition(image: np.ndarray, condition: str, sigma: int = 10) -> np.ndarray:
    """Return a new RGB uint8 image using the frozen Member 3 parameters."""
```

Use `add_gaussian_noise`, `bilateral_filter_luminance`, `agcwd_luminance`, and `apply_member3_pipeline` from `member3.py`. Map sigma to the existing `NOISE_SEEDS`; reject sigma values outside `(10, 25, 40)`. Keep the input untouched with `image.copy()` for `Clean`.

- [ ] **Step 4: Run the focused tests and verify they pass.**

Run the same pytest command. Expected: all condition tests pass.

- [ ] **Step 5: Commit the first vertical slice.**

```bash
git add src/hripcb_baseline/member3_demo.py tests/test_member3_demo.py
git commit -m "Add Member 3 demo preprocessing seam"
```

### Task 2: Add prediction, annotations, and optional ground truth

**Files:**
- Modify: `src/hripcb_baseline/member3_demo.py`
- Modify: `tests/test_member3_demo.py`

**Interfaces:**
- Consumes: RGB `np.ndarray` from `prepare_condition`; an Ultralytics-compatible model or test double.
- Produces: serialisable detection records, annotated RGB images, and optional ground-truth records.

- [ ] **Step 1: Write the failing prediction and annotation tests.**

Use a fake result object exposing `boxes.xyxy`, `boxes.conf`, and `boxes.cls`, plus a fake model whose `predict` method records `imgsz`, `conf`, and `verbose` and returns one result. Assert `predict_image` returns records with `class_name`, `class_id`, `confidence`, and integer `xyxy`. Assert `draw_detections` preserves image dimensions and changes pixels around the supplied box.

```python
def test_predict_image_uses_fixed_inference_size_and_serialises_boxes():
    model = FakeModel([(12.2, 8.8, 40.9, 31.4, 0.87, 2)])
    records = predict_image(model, image, conf=0.25, imgsz=1024)
    assert records == [{
        "class_id": 2,
        "class_name": "Open_circuit",
        "confidence": 0.87,
        "xyxy": [12, 9, 41, 31],
    }]
    assert model.last_kwargs == {"imgsz": 1024, "conf": 0.25, "verbose": False}
```

Also test YOLO's empty-box result produces an empty list, and test a label line `2 0.5 0.5 0.2 0.4` is converted to a box using the provided image width and height.

- [ ] **Step 2: Run the focused tests and verify they fail.**

```bash
python3 -m pytest tests/test_member3_demo.py -q
```

Expected: failures for the missing `predict_image`, `draw_detections`, and `load_ground_truth` seams.

- [ ] **Step 3: Implement model adapter, annotation, and label loading.**

Define these exact public functions: `predict_image(model: Any, image: np.ndarray, *, conf: float = 0.25, imgsz: int = 1024) -> list[dict[str, object]]`, `draw_detections(image: np.ndarray, detections: Sequence[Mapping[str, object]], *, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray`, and `load_ground_truth(label_path: Path, image_shape: tuple[int, int, int]) -> list[dict[str, object]]`.

Call `model.predict(source=image, imgsz=imgsz, conf=conf, verbose=False)`. Convert tensors through `.cpu().numpy()` when available, then zip boxes, confidences, and classes. Use `cv2.rectangle` and `cv2.putText` for predictions. Parse YOLO normalized labels into clipped pixel `xyxy` boxes and skip malformed lines without crashing.

- [ ] **Step 4: Run the focused tests and verify they pass.**

```bash
python3 -m pytest tests/test_member3_demo.py -q
```

Expected: all preprocessing, prediction, annotation, and ground-truth tests pass.

- [ ] **Step 5: Commit the second vertical slice.**

```bash
git add src/hripcb_baseline/member3_demo.py tests/test_member3_demo.py
git commit -m "Add Member 3 demo prediction and annotation seams"
```

### Task 3: Add artefact persistence and CSV summary loading

**Files:**
- Modify: `src/hripcb_baseline/member3_demo.py`
- Modify: `tests/test_member3_demo.py`

**Interfaces:**
- Consumes: original/processed/annotated RGB arrays, detection records, and `runs/member3/comparison.csv`.
- Produces: timestamped output directory and safe summary rows for Streamlit.

- [ ] **Step 1: Write failing tests for save/load behavior.**

Use `tmp_path` to assert `save_demo_artifacts` creates `original.png`, `processed.png`, `prediction.png`, and `metadata.json`. Assert metadata contains `condition`, `sigma`, `confidence_threshold`, `model_path`, and `detections`. Write a two-row CSV fixture and assert `load_summary_rows` returns only rows with `split == "test"` when requested.

- [ ] **Step 2: Run the focused tests and verify they fail.**

```bash
python3 -m pytest tests/test_member3_demo.py -q
```

Expected: failures for the missing persistence and CSV seams.

- [ ] **Step 3: Implement persistence and summary loading.**

Define these exact public functions: `save_demo_artifacts(output_root: Path, *, original: np.ndarray, processed: np.ndarray, prediction: np.ndarray, metadata: Mapping[str, object], source_name: str) -> Path` and `load_summary_rows(csv_path: Path, *, split: str | None = "test") -> list[dict[str, str]]`.

Create a `YYYYMMDD-HHMMSS` directory with a collision-safe suffix, save RGB arrays using Pillow, and serialize metadata with `json.dump(metadata, handle, indent=2, default=str)`. Use `csv.DictReader`; preserve the existing CSV column names and filter `split` only when requested. Missing CSV returns an empty list.

- [ ] **Step 4: Run the focused tests and verify they pass.**

```bash
python3 -m pytest tests/test_member3_demo.py -q
```

Expected: all demo module tests pass.

- [ ] **Step 5: Commit the third vertical slice.**

```bash
git add src/hripcb_baseline/member3_demo.py tests/test_member3_demo.py
git commit -m "Add Member 3 demo artefact and summary helpers"
```

### Task 4: Build the Streamlit dashboard

**Files:**
- Create: `scripts/member3_demo.py`
- Modify: `requirements.txt`
- Modify: `tests/test_member3_demo.py`

**Interfaces:**
- Consumes: the public seams from `member3_demo.py`, `runs/baseline/weights/best.pt`, optional dataset root, and `runs/member3/comparison.csv`.
- Produces: a local dashboard runnable with `streamlit run scripts/member3_demo.py`.

- [ ] **Step 1: Add a smoke test before the UI implementation.**

Add `test_demo_script_compiles` that runs `py_compile.compile` on the script, and add `streamlit>=1.40,<2` to `requirements.txt`. Do not import Streamlit in unit tests so the pure seams remain testable without a browser.

- [ ] **Step 2: Run the smoke test and verify it fails.**

```bash
python3 -m pytest tests/test_member3_demo.py::test_demo_script_compiles -q
```

Expected: the script path does not exist yet.

- [ ] **Step 3: Implement the dashboard.**

The script must:

1. Set the page title and show `Member 3 AI PCB Defect Inspector`.
2. Define sidebar controls for upload, condition, sigma, confidence slider (`0.05` to `0.90`, default `0.25`), model path, and optional dataset root.
3. Load the model with `@st.cache_resource`, validate that the checkpoint exists, and use automatic device selection through Ultralytics.
4. Convert the upload to RGB NumPy, call `prepare_condition`, `predict_image`, and `draw_detections`, and show Original/Processed/Detection Result in columns.
5. Match an uploaded filename against `train`, `val`, or `test` image folders under the optional dataset root; load the matching label file only when it exists and show Ground Truth separately.
6. Show detection rows with class, confidence, and coordinates, plus count and inference duration.
7. Implement `Compare All Conditions` by applying all five condition labels to the same image and rendering one annotated result per condition. Use the selected sigma for noise-bearing conditions.
8. Save each completed run with `save_demo_artifacts` and show the output path.
9. Load `runs/member3/comparison.csv`, filter to test rows, format `map50_95` as a percentage, and show it under `Experiment Summary`. If absent, show an informational message rather than failing.
10. Catch missing files, invalid uploads, and inference exceptions with `st.error`/`st.warning` and continue rendering the rest of the page.

- [ ] **Step 4: Install the new dependency and run the smoke test.**

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest tests/test_member3_demo.py -q
python3 -m py_compile scripts/member3_demo.py src/hripcb_baseline/member3_demo.py
```

Expected: all demo tests pass and both Python files compile.

- [ ] **Step 5: Commit the dashboard slice.**

```bash
git add requirements.txt scripts/member3_demo.py src/hripcb_baseline/member3_demo.py tests/test_member3_demo.py
git commit -m "Add Streamlit Member 3 demo dashboard"
```

### Task 5: Document usage and verify the complete feature

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the completed dashboard command and output locations.
- Produces: a reproducible usage section and verification evidence.

- [ ] **Step 1: Add the README usage section.**

Document:

```bash
python3 -m pip install -r requirements.txt
streamlit run scripts/member3_demo.py
```

Explain that the app uses the frozen `best.pt`, that the CSV is a batch metric summary, and that interactive single-image results are saved under `runs/member3_demo/`.

- [ ] **Step 2: Run the focused and existing tests.**

```bash
python3 -m pytest tests/test_member3_demo.py tests/test_member3.py tests/test_member3_experiment.py tests/test_member3_runner.py tests/test_member3_metrics.py -q
```

Expected: all new and Member 3 tests pass. If the pre-existing external-dataset configuration test is run as part of the full suite, report that known failure separately rather than changing dataset paths.

- [ ] **Step 3: Run the final static checks.**

```bash
python3 -m py_compile scripts/member3_demo.py src/hripcb_baseline/member3_demo.py
git diff --check HEAD~5..HEAD
git status --short
```

Expected: compilation succeeds, no whitespace errors are introduced by the feature, and only intended source/documentation changes remain; generated `runs/member3` output remains uncommitted.

- [ ] **Step 4: Perform a local launch smoke check.**

Run:

```bash
python3 -m streamlit run scripts/member3_demo.py --server.headless true
```

Open `http://localhost:8501`, upload one dataset image, run one condition, verify the three image panels and detection table, then stop the server with `Control+C`. Also verify a new directory exists below `runs/member3_demo/`.

- [ ] **Step 5: Commit documentation and final verified feature.**

```bash
git add README.md requirements.txt scripts/member3_demo.py src/hripcb_baseline/member3_demo.py tests/test_member3_demo.py
git commit -m "Document and verify Member 3 interactive demo"
```
