# Shared YOLOv8s PCB Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate one reproducible YOLOv8s checkpoint for all later HRIPCB preprocessing experiments.

**Architecture:** A small Python package validates the existing YOLO dataset and provides stable training/evaluation entry points. The local dataset configuration is generated from the repository root, while Ultralytics owns model training and metric calculation. All later experiments consume the saved `best.pt` checkpoint without retraining it.

**Tech Stack:** Python 3.13, Ultralytics YOLOv8, PyTorch MPS, OpenCV/Pillow, PyYAML, pytest.

## Global Constraints

- Preserve the supplied `HRIPCB_UPDATE/train`, `val`, and `test` split for the first baseline.
- Use six classes in this exact order: `Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spurious_copper`, `Spur`.
- Use clean images for baseline training; degradation and specialized preprocessing are later inference-only experiments.
- Use seed 42, image size 1024, maximum 100 epochs, patience 20, batch size 4 initially, and MPS when available.
- Do not overwrite the original dataset images or labels.
- Every later member must reuse the same `runs/baseline/weights/best.pt` checkpoint.

---

### Task 1: Create the Python project and dataset-validation contract

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/hripcb_baseline/__init__.py`
- Create: `src/hripcb_baseline/dataset.py`
- Create: `tests/test_dataset.py`
- Create: `scripts/validate_dataset.py`

**Interfaces:**
- `hripcb_baseline.dataset.validate_dataset(root: Path) -> DatasetReport`
- `DatasetReport.total_images: int`
- `DatasetReport.total_labels: int`
- `DatasetReport.split_counts: dict[str, int]`
- `DatasetReport.class_counts: dict[int, int]`
- `DatasetReport.errors: list[str]`
- `DatasetReport.ok: bool`

- [x] **Step 1: Write failing tests**

```python
def test_validate_dataset_reports_expected_split_counts(tmp_path):
    # The test fixture contains one valid image/label pair per split.
    report = validate_dataset(tmp_path)
    assert report.total_images == 3
    assert report.total_labels == 3
    assert report.split_counts == {"train": 1, "val": 1, "test": 1}
    assert report.ok is True


def test_validate_dataset_rejects_invalid_class_and_box(tmp_path):
    # A label with class 6 and an x-center outside [0, 1] must fail validation.
    report = validate_dataset(tmp_path)
    assert report.ok is False
    assert any("class" in error for error in report.errors)
    assert any("normalized" in error for error in report.errors)
```

- [x] **Step 2: Run the tests and verify the expected failure**

Run: `python3 -m pytest tests/test_dataset.py -q`

Expected: FAIL because `hripcb_baseline.dataset` and `validate_dataset` do not exist yet.

- [x] **Step 3: Implement the validator**

Implement the exact `DatasetReport` and `validate_dataset` interface. For each of `train`, `val`, and `test`, inspect `images/*` and require the same-stem `labels/*.txt`. Parse each non-empty label line as five numeric values, require class IDs 0–5, and require normalized center/width/height values in [0, 1] with positive width and height. Read images with Pillow to detect unreadable files. Do not modify files.

- [x] **Step 4: Add the CLI**

`python scripts/validate_dataset.py --root HRIPCB_UPDATE` must print split counts, class counts, and all errors, then exit 0 only when `report.ok` is true.

- [x] **Step 5: Run tests and validate the real dataset**

Run:

```bash
python3 -m pytest tests/test_dataset.py -q
python3 scripts/validate_dataset.py --root HRIPCB_UPDATE
```

Expected: tests pass; the real dataset reports 693 images, 693 labels, and no validation errors.

---

### Task 2: Add a local YOLO dataset configuration

**Files:**
- Create: `configs/hripcb_local.yaml`
- Create: `src/hripcb_baseline/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- `hripcb_baseline.config.load_config(path: Path) -> dict`
- YAML keys `path`, `train`, `val`, `test`, `nc`, and `names`.
- `path` must resolve to the absolute local `HRIPCB_UPDATE` directory at runtime or be a project-relative path accepted by Ultralytics.

- [x] **Step 1: Write failing configuration tests**

```python
def test_local_dataset_config_has_six_expected_classes():
    config = load_config(Path("configs/hripcb_local.yaml"))
    assert config["nc"] == 6
    assert config["names"] == [
        "Missing_hole", "Mouse_bite", "Open_circuit",
        "Short", "Spurious_copper", "Spur",
    ]


def test_local_dataset_config_points_to_existing_split_directories():
    config = load_config(Path("configs/hripcb_local.yaml"))
    root = Path(config["path"])
    assert (root / config["train"] / "images").is_dir()
    assert (root / config["val"] / "images").is_dir()
    assert (root / config["test"] / "images").is_dir()
```

- [x] **Step 2: Run the tests and verify failure**

Run: `python3 -m pytest tests/test_config.py -q`

Expected: FAIL because the local configuration does not exist.

- [x] **Step 3: Implement the local configuration**

Use a project-relative path:

```yaml
path: HRIPCB_UPDATE
train: train
val: val
test: test
nc: 6
names:
  0: Missing_hole
  1: Mouse_bite
  2: Open_circuit
  3: Short
  4: Spurious_copper
  5: Spur
```

- [x] **Step 4: Run configuration tests**

Run: `python3 -m pytest tests/test_config.py -q`

Expected: PASS.

---

### Task 3: Create the shared YOLOv8s training entry point

**Files:**
- Create: `configs/baseline.yaml`
- Create: `scripts/train_baseline.py`
- Create: `tests/test_training_config.py`
- Modify: `requirements.txt`

**Interfaces:**
- `python scripts/train_baseline.py --data configs/hripcb_local.yaml --model yolov8s.pt --project runs --name baseline`
- The command must create `runs/baseline/weights/best.pt` after successful full training.

- [x] **Step 1: Write failing training-configuration tests**

```python
def test_baseline_config_is_shared_and_reproducible():
    config = load_config(Path("configs/baseline.yaml"))
    assert config["model"] == "yolov8s.pt"
    assert config["imgsz"] == 1024
    assert config["seed"] == 42
    assert config["epochs"] == 100
    assert config["patience"] == 20
    assert config["batch"] == 4
```

- [x] **Step 2: Run the test and verify failure**

Run: `python3 -m pytest tests/test_training_config.py -q`

Expected: FAIL because `configs/baseline.yaml` does not exist.

- [x] **Step 3: Implement the fixed baseline configuration and script**

The script must load Ultralytics `YOLO`, select `mps` when `torch.backends.mps.is_available()` is true, otherwise use CPU, and pass explicit `data`, `imgsz=1024`, `epochs=100`, `patience=20`, `batch=4`, `seed=42`, `workers=0`, `project="runs"`, and `name="baseline"`. Keep training on clean images and leave test evaluation to the evaluation entry point. Make the model path and batch size overridable from the CLI without changing the recorded defaults.

- [x] **Step 4: Install dependencies and run a smoke test**

Run:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/train_baseline.py --epochs 1 --imgsz 640 --batch 2 --name baseline_smoke
```

Expected: one epoch completes and creates a smoke-run directory with a checkpoint or training results. This smoke test is not the shared final model.

- [x] **Step 5: Run the full shared training**

Run:

```bash
python3 scripts/train_baseline.py --data configs/hripcb_local.yaml --model yolov8s.pt --project runs --name baseline
```

Expected: `runs/baseline/weights/best.pt` exists and the run records its arguments and metrics.

---

### Task 4: Evaluate and freeze the shared checkpoint

**Files:**
- Create: `scripts/evaluate_baseline.py`
- Create: `tests/test_evaluation_config.py`
- Create: `README.md`

**Interfaces:**
- `python scripts/evaluate_baseline.py --weights runs/baseline/weights/best.pt --data configs/hripcb_local.yaml --split val`
- `python scripts/evaluate_baseline.py --weights runs/baseline/weights/best.pt --data configs/hripcb_local.yaml --split test`
- The evaluator must write metrics and preserve the split name in the output directory.
- `scripts/evaluate_baseline.py` must return a non-zero exit code and write a checkpoint-related error to stderr when the weights path is missing.

- [x] **Step 1: Write failing evaluator tests**

```python
def test_evaluation_requires_a_checkpoint(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_baseline.py",
         "--weights", str(tmp_path / "missing.pt"), "--split", "test"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "checkpoint" in result.stderr.lower()
```

- [x] **Step 2: Run the test and verify failure**

Run: `python3 -m pytest tests/test_evaluation_config.py -q`

Expected: FAIL because the evaluator does not exist yet.

- [x] **Step 3: Implement evaluation**

Load the fixed checkpoint, call Ultralytics validation with the selected split, `imgsz=1024`, `conf=0.25`, `iou=0.7`, `device` selected in the same way as training, and save results under `runs/evaluation/<split>`. Export the metrics needed later: mAP@0.5, mAP@0.5:0.95, precision, recall, F1, and per-class AP where available.

- [x] **Step 4: Run clean validation and test evaluation**

Run:

```bash
python3 scripts/evaluate_baseline.py --weights runs/baseline/weights/best.pt --data configs/hripcb_local.yaml --split val
python3 scripts/evaluate_baseline.py --weights runs/baseline/weights/best.pt --data configs/hripcb_local.yaml --split test
```

Expected: both commands complete and produce metrics for all six classes.

- [x] **Step 5: Document the shared checkpoint contract**

Document the exact checkpoint path, dataset split, class order, model settings, and commands that every member must use. State that later denoising/contrast experiments must not retrain or replace this checkpoint.

---

### Task 5: Final verification and handoff

**Files:**
- Modify: `README.md`
- Create: `artifacts/baseline-manifest.json`

- [x] **Step 1: Run the complete test suite**

Run: `python3 -m pytest -q`

Expected: all project tests pass.

- [x] **Step 2: Verify the checkpoint and outputs**

Run:

```bash
test -f runs/baseline/weights/best.pt
python3 scripts/validate_dataset.py --root HRIPCB_UPDATE
```

Expected: checkpoint exists and dataset validation exits successfully.

- [x] **Step 3: Write the manifest**

Record the checkpoint path, dataset counts, class order, image size, seed, epochs, device, package versions, and validation/test result locations in `artifacts/baseline-manifest.json` without copying the dataset or model weights into source control.

- [x] **Step 4: Handoff**

The handoff consists of the shared checkpoint path, reproducible commands, clean baseline metrics, and the explicit limitation that this is based on the supplied split rather than a verified template-board group split.
