# Member 5 TV + Top-hat/Black-hat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fully integrated Member 5 TV plus Top-hat/Black-hat preprocessing module and a resumable, disk-bounded 52-candidate validation search that the user runs manually.

**Architecture:** Keep image math in the shared preprocessing filters, candidate serialization and order semantics in the shared candidate dispatcher, and batch lifecycle safety in a small incremental-run helper. A dedicated Member 5 CLI evaluates explicit candidate batches, atomically checkpoints completed records, deletes successful batch datasets, and publishes complete results to the existing five-member dashboard only after all 52 candidates finish.

**Tech Stack:** Python 3.10+, NumPy 2.x, OpenCV 4.10+, scikit-image 0.26, PyYAML, Ultralytics YOLO, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-member5-tv-morphology-design.md`

## Global Constraints

- Member 5 is TV denoising followed by one enhancement stage that computes Top-hat and Black-hat independently from the same base image.
- Use `denoise_tv_chambolle`; tune only `weight`, with `eps=0.0002`, `max_num_iter=200`, and `channel_axis=-1` fixed.
- Use TV weights `[0.01, 0.02, 0.05]`, shared elliptical kernel sizes `[5, 9, 15]`, Top-hat amounts `[0.5, 1.0]`, and Black-hat amounts `[0.5, 1.0]`.
- Produce exactly 1 original, 3 TV-only, 12 morphology-only, and 36 mandatory combined candidates: 52 total.
- Select the winner only from `tv_top_black_hat` records on `val`, ranked by `mAP50-95`.
- Preserve the frozen checkpoint, `imgsz=1024`, `conf=0.25`, `iou=0.70`, `workers=0`, and all Member 1-4 preprocessing behavior.
- Persist completed records before deleting batch images. Partial results must never enter the project-wide leaderboard.
- Do not add Member 5 to `scripts/run_all_validation_sweeps.py`; its dedicated full-search script is intentionally run manually.
- Do not run the 52-candidate sweep during implementation or verification.
- Do not commit, push, deploy, retrain, or evaluate the test split as part of running the search script.

---

## File Map

- Modify `src/hripcb_preprocessing/filters.py`: TV and luminance morphology primitives.
- Modify `src/hripcb_preprocessing/candidates.py`: Member 5 grid and dispatch.
- Create `configs/member5_full_search.yaml`: approved 52-candidate protocol.
- Modify `src/hripcb_preprocessing/runner.py`: accept an explicit candidate batch while retaining existing behavior.
- Create `src/hripcb_preprocessing/incremental.py`: fingerprint, atomic progress, and guarded batch cleanup.
- Create `scripts/run_member5_full_search.py`: batching, resume, final summary, and final project aggregation.
- Modify `scripts/build_project_dashboard.py`: recognize the Member 5 completed-result source.
- Modify `src/hripcb_dashboard/filtering.py`: fifth module and combined identifier.
- Modify `src/hripcb_dashboard/analysis.py`: labels and Member 5 stage analysis.
- Modify `src/hripcb_dashboard/dashboard.py`: static dashboard combined/shared-control constants.
- Modify `scripts/streamlit_dashboard.py`: five-member wording and Member 5 winner callout.
- Modify `scripts/run_preprocessing_sweep.py`: remove obsolete Member 1-4 wording from the generic CLI.
- Modify `README.md`: Member 5 method and manual-run instructions.
- Modify `tests/test_preprocessing_modules.py`: filter, grid, and order behavior.
- Create `tests/test_member5_full_search.py`: resume, cleanup, summary, and no-partial-publication behavior.
- Modify `tests/test_preprocessing_runner_resume.py`: explicit batch and lifecycle helper behavior.
- Modify `tests/test_dashboard_filtering.py`, `tests/test_dashboard_extra_effort.py`, `tests/test_project_dashboard_sorting.py`, and `tests/test_experiment_dashboard.py`: five-member UI and aggregation coverage.

### Task 1: TV and Top-hat/Black-hat Filter Primitives

**Files:**
- Modify: `tests/test_preprocessing_modules.py`
- Modify: `src/hripcb_preprocessing/filters.py`

**Interfaces:**
- Produces: `apply_tv_denoise(image: np.ndarray, weight: float = 0.02) -> np.ndarray`
- Produces: `apply_top_black_hat(image: np.ndarray, kernel_size: int = 9, top_amount: float = 0.5, black_amount: float = 0.5) -> np.ndarray`
- Depends on: existing `_validate_image(image)`.

- [ ] **Step 1: Write failing TV behavior and validation tests**

Add the two new imports and these tests to `tests/test_preprocessing_modules.py`:

```python
from hripcb_preprocessing.filters import (
    apply_homomorphic_filter,
    apply_top_black_hat,
    apply_tv_denoise,
    apply_wavelet_denoise,
)


def test_tv_denoise_preserves_shape_dtype_and_does_not_mutate_input():
    image = _image()
    before = image.copy()

    actual = apply_tv_denoise(image, weight=0.02)

    assert actual.shape == image.shape
    assert actual.dtype == np.uint8
    assert np.array_equal(image, before)


@pytest.mark.parametrize("weight", [0.0, -0.01])
def test_tv_denoise_rejects_non_positive_weight(weight):
    with pytest.raises(ValueError, match="weight"):
        apply_tv_denoise(_image(), weight=weight)
```

- [ ] **Step 2: Write failing morphology equation and validation tests**

```python
def test_top_black_hat_matches_same_input_luminance_equation():
    image = _image()
    kernel_size = 5
    top_amount = 0.5
    black_amount = 1.0
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    base = ycrcb[..., 0]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    top = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, kernel)
    black = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, kernel)
    expected_y = np.clip(
        base.astype(np.float32)
        + top_amount * top.astype(np.float32)
        - black_amount * black.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    expected_ycrcb = ycrcb.copy()
    expected_ycrcb[..., 0] = expected_y
    expected = cv2.cvtColor(expected_ycrcb, cv2.COLOR_YCrCb2BGR)

    actual = apply_top_black_hat(
        image,
        kernel_size=kernel_size,
        top_amount=top_amount,
        black_amount=black_amount,
    )

    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("kernel_size", [0, 4, -3])
def test_top_black_hat_rejects_non_positive_or_even_kernel(kernel_size):
    with pytest.raises(ValueError, match="kernel_size"):
        apply_top_black_hat(_image(), kernel_size=kernel_size)


@pytest.mark.parametrize(
    ("top_amount", "black_amount"),
    [(-0.1, 0.5), (0.5, -0.1)],
)
def test_top_black_hat_rejects_negative_amounts(top_amount, black_amount):
    with pytest.raises(ValueError, match="amount"):
        apply_top_black_hat(
            _image(), top_amount=top_amount, black_amount=black_amount
        )
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_preprocessing_modules.py \
  -k 'tv_denoise or top_black_hat'
```

Expected: test collection fails because the two filter functions do not exist.

- [ ] **Step 4: Implement the two filter primitives**

Add `denoise_tv_chambolle` to the existing restoration import and add these functions after `_validate_image`:

```python
def apply_tv_denoise(
    image: np.ndarray,
    weight: float = 0.02,
) -> np.ndarray:
    """Denoise a colour image with Chambolle total variation."""

    _validate_image(image)
    if weight <= 0:
        raise ValueError("weight must be positive")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    denoised = denoise_tv_chambolle(
        rgb,
        weight=float(weight),
        eps=0.0002,
        max_num_iter=200,
        channel_axis=-1,
    )
    rgb_uint8 = np.clip(np.rint(denoised * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)


def apply_top_black_hat(
    image: np.ndarray,
    kernel_size: int = 9,
    top_amount: float = 0.5,
    black_amount: float = 0.5,
) -> np.ndarray:
    """Enhance small bright and dark luminance structures morphologically."""

    _validate_image(image)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if top_amount < 0 or black_amount < 0:
        raise ValueError("top_amount and black_amount must not be negative")
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[..., 0]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(kernel_size), int(kernel_size))
    )
    top = cv2.morphologyEx(luminance, cv2.MORPH_TOPHAT, kernel)
    black = cv2.morphologyEx(luminance, cv2.MORPH_BLACKHAT, kernel)
    enhanced = (
        luminance.astype(np.float32)
        + float(top_amount) * top.astype(np.float32)
        - float(black_amount) * black.astype(np.float32)
    )
    result = ycrcb.copy()
    result[..., 0] = np.clip(enhanced, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)
```

- [ ] **Step 5: Run focused and existing preprocessing tests**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_preprocessing_modules.py
```

Expected: all tests in the file pass.

- [ ] **Step 6: Commit the filter primitives**

```bash
git add src/hripcb_preprocessing/filters.py tests/test_preprocessing_modules.py
git commit -m "feat: add Member 5 TV morphology filters"
```

### Task 2: Member 5 Candidate Grid, Dispatch, and Configuration

**Files:**
- Modify: `tests/test_preprocessing_modules.py`
- Modify: `src/hripcb_preprocessing/candidates.py`
- Create: `configs/member5_full_search.yaml`

**Interfaces:**
- Consumes: `apply_tv_denoise(...)` and `apply_top_black_hat(...)` from Task 1.
- Produces: `build_candidates("member5", config) -> list[dict]` with 52 records.
- Produces technique identifiers: `tv`, `top_black_hat`, `tv_top_black_hat`.

- [ ] **Step 1: Add a repository-config grid test**

```python
from pathlib import Path

import yaml


def test_member5_repository_config_builds_52_candidates():
    config = yaml.safe_load(
        Path("configs/member5_full_search.yaml").read_text(encoding="utf-8")
    )

    candidates = build_candidates("member5", config)

    assert len(candidates) == 52
    assert sum(row["technique"] == "original" for row in candidates) == 1
    assert sum(row["technique"] == "tv" for row in candidates) == 3
    assert sum(row["technique"] == "top_black_hat" for row in candidates) == 12
    assert sum(row["technique"] == "tv_top_black_hat" for row in candidates) == 36
    assert len({row["id"] for row in candidates}) == 52
```

- [ ] **Step 2: Add combined-parameter and exact-order tests**

```python
def test_member5_combined_candidates_use_all_required_operations():
    config = {
        "tv_weights": [0.02],
        "morphology_kernel_sizes": [9],
        "top_hat_amounts": [0.5],
        "black_hat_amounts": [1.0],
    }
    combined = next(
        row
        for row in build_candidates("member5", config)
        if row["technique"] == "tv_top_black_hat"
    )

    assert combined["parameters"] == {
        "tv_weight": 0.02,
        "morphology_kernel_size": 9,
        "top_hat_amount": 0.5,
        "black_hat_amount": 1.0,
    }


def test_member5_combined_dispatch_is_tv_then_same_input_morphology():
    image = _image()
    parameters = {
        "tv_weight": 0.02,
        "morphology_kernel_size": 5,
        "top_hat_amount": 0.5,
        "black_hat_amount": 1.0,
    }
    candidate = {
        "module": "member5",
        "technique": "tv_top_black_hat",
        "parameters": parameters,
    }
    expected = apply_top_black_hat(
        apply_tv_denoise(image, weight=0.02),
        kernel_size=5,
        top_amount=0.5,
        black_amount=1.0,
    )

    assert np.array_equal(apply_candidate(image, candidate), expected)
```

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_preprocessing_modules.py -k member5
```

Expected: failures because the config and Member 5 dispatcher are absent.

- [ ] **Step 4: Create the approved YAML configuration**

Create `configs/member5_full_search.yaml` with:

```yaml
module: member5
dataset: HRIPCB_UPDATE
split: val
checkpoint: runs/baseline/weights/best.pt
data_config: configs/hripcb_local.yaml
imgsz: 1024
conf: 0.25
iou: 0.7
device: auto
workers: 0
jpeg_quality: 95
image_metrics_max_side: 256
primary_metric: map50_95
model_id: baseline
model_label: Baseline YOLO
training_preprocessing: original
evaluation_type: ablation
tv_weights: [0.01, 0.02, 0.05]
morphology_kernel_sizes: [5, 9, 15]
top_hat_amounts: [0.5, 1.0]
black_hat_amounts: [0.5, 1.0]
```

- [ ] **Step 5: Add Member 5 construction and dispatch**

Import the two Task 1 functions in `candidates.py`. Add a `member5` branch before
the unsupported-module error. Generate single controls and the full Cartesian
product with IDs that encode every parameter:

```python
if module == "member5":
    weights = [float(value) for value in config["tv_weights"]]
    kernels = [int(value) for value in config["morphology_kernel_sizes"]]
    top_amounts = [float(value) for value in config["top_hat_amounts"]]
    black_amounts = [float(value) for value in config["black_hat_amounts"]]
    for weight in weights:
        candidates.append(_candidate(
            f"tv_w{_label(weight)}", module, "tv", {"tv_weight": weight}
        ))
    for kernel in kernels:
        for top_amount in top_amounts:
            for black_amount in black_amounts:
                morphology = {
                    "morphology_kernel_size": kernel,
                    "top_hat_amount": top_amount,
                    "black_hat_amount": black_amount,
                }
                suffix = (
                    f"k{kernel}_t{_label(top_amount)}_b{_label(black_amount)}"
                )
                candidates.append(_candidate(
                    f"top_black_hat_{suffix}",
                    module,
                    "top_black_hat",
                    morphology,
                ))
                for weight in weights:
                    candidates.append(_candidate(
                        f"tv_w{_label(weight)}_top_black_hat_{suffix}",
                        module,
                        "tv_top_black_hat",
                        {"tv_weight": weight, **morphology},
                    ))
    return candidates
```

Add these dispatch branches before the final error:

```python
if technique == "tv":
    return apply_tv_denoise(image, float(parameters["tv_weight"]))
if technique == "top_black_hat":
    return apply_top_black_hat(
        image,
        int(parameters["morphology_kernel_size"]),
        float(parameters["top_hat_amount"]),
        float(parameters["black_hat_amount"]),
    )
if technique == "tv_top_black_hat":
    denoised = apply_tv_denoise(image, float(parameters["tv_weight"]))
    return apply_top_black_hat(
        denoised,
        int(parameters["morphology_kernel_size"]),
        float(parameters["top_hat_amount"]),
        float(parameters["black_hat_amount"]),
    )
```

- [ ] **Step 6: Run focused and regression tests**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_preprocessing_modules.py
```

Expected: all preprocessing-module tests pass and the configured counts are
exactly 52 total and 36 combined.

- [ ] **Step 7: Commit the candidate grid**

```bash
git add configs/member5_full_search.yaml \
  src/hripcb_preprocessing/candidates.py \
  tests/test_preprocessing_modules.py
git commit -m "feat: define Member 5 candidate matrix"
```

### Task 3: Explicit Candidate Batches and Safe Progress Primitives

**Files:**
- Modify: `src/hripcb_preprocessing/runner.py`
- Create: `src/hripcb_preprocessing/incremental.py`
- Modify: `tests/test_preprocessing_runner_resume.py`

**Interfaces:**
- Produces: `run_sweep(dataset_root, output_root, config, *, candidates=None) -> Path`.
- Produces: `config_fingerprint(config: Mapping, dataset_root: Path) -> str`.
- Produces: `load_progress(path: Path, expected_fingerprint: str) -> dict`.
- Produces: `atomic_write_json(path: Path, payload: object) -> None`.
- Produces: `remove_completed_batch(batch_root: Path, batches_root: Path) -> None`.

- [ ] **Step 1: Add failing tests for explicit candidate batches**

In `tests/test_preprocessing_runner_resume.py`, import `_resolve_candidates` and
add:

```python
from hripcb_preprocessing.runner import _resolve_candidates


def test_resolve_candidates_uses_explicit_batch_without_rebuilding(monkeypatch):
    explicit = [{"id": "only", "module": "member5", "technique": "original", "parameters": {}}]

    def fail_build(*_args, **_kwargs):
        raise AssertionError("full grid must not be rebuilt")

    monkeypatch.setattr("hripcb_preprocessing.runner.build_candidates", fail_build)

    assert _resolve_candidates("member5", {}, explicit) == explicit
```

- [ ] **Step 2: Add failing fingerprint, atomic-write, and cleanup tests**

```python
import json

from hripcb_preprocessing.incremental import (
    atomic_write_json,
    config_fingerprint,
    load_progress,
    remove_completed_batch,
)


def test_progress_rejects_a_changed_configuration(tmp_path):
    path = tmp_path / "progress.json"
    atomic_write_json(path, {
        "version": 1,
        "fingerprint": "first",
        "records": [],
    })

    with pytest.raises(ValueError, match="configuration"):
        load_progress(path, "second")


def test_atomic_write_json_replaces_complete_payload(tmp_path):
    path = tmp_path / "progress.json"
    atomic_write_json(path, {"records": [{"id": "one"}]})
    atomic_write_json(path, {"records": [{"id": "one"}, {"id": "two"}]})

    assert [row["id"] for row in json.loads(path.read_text())["records"]] == [
        "one", "two"
    ]
    assert not path.with_suffix(".json.tmp").exists()


def test_remove_completed_batch_refuses_paths_outside_batch_root(tmp_path):
    batches_root = tmp_path / "output" / "_batches"
    allowed = batches_root / "batch_001_008"
    allowed.mkdir(parents=True)
    remove_completed_batch(allowed, batches_root)
    assert not allowed.exists()

    outside = tmp_path / "keep"
    outside.mkdir()
    with pytest.raises(ValueError, match="batch root"):
        remove_completed_batch(outside, batches_root)
    assert outside.exists()
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_preprocessing_runner_resume.py
```

Expected: import failures for the new helper module and resolver.

- [ ] **Step 4: Add explicit-candidate support without changing callers**

In `runner.py`, add:

```python
def _resolve_candidates(
    module: str,
    config: dict,
    candidates: list[dict] | None,
) -> list[dict]:
    return build_candidates(module, config) if candidates is None else list(candidates)
```

Change the public signature and the candidate assignment:

```python
def run_sweep(
    dataset_root: Path,
    output_root: Path,
    config: dict,
    *,
    candidates: list[dict] | None = None,
) -> Path:
    # existing path setup remains unchanged
    module = str(config["module"])
    candidates = _resolve_candidates(module, config, candidates)
```

All existing positional callers continue to build their complete grids.

- [ ] **Step 5: Implement the incremental lifecycle helpers**

Create `src/hripcb_preprocessing/incremental.py`:

```python
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path


def config_fingerprint(config: Mapping, dataset_root: Path) -> str:
    payload = {
        "config": dict(config),
        "dataset_root": str(Path(dataset_root).resolve()),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_progress(path: Path, expected_fingerprint: str) -> dict:
    path = Path(path)
    if not path.is_file():
        return {
            "version": 1,
            "fingerprint": expected_fingerprint,
            "records": [],
        }
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "Existing progress belongs to a different configuration; "
            "choose a new output directory"
        )
    return progress


def remove_completed_batch(batch_root: Path, batches_root: Path) -> None:
    batch_root = Path(batch_root).resolve()
    batches_root = Path(batches_root).resolve()
    try:
        relative = batch_root.relative_to(batches_root)
    except ValueError as error:
        raise ValueError("cleanup target is outside the batch root") from error
    if not relative.parts or batch_root == batches_root:
        raise ValueError("cleanup target must be a child of the batch root")
    if batch_root.exists():
        shutil.rmtree(batch_root)
```

- [ ] **Step 6: Run runner-focused regression tests**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_preprocessing_runner_resume.py \
  tests/test_preprocessing_modules.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the batching primitives**

```bash
git add src/hripcb_preprocessing/runner.py \
  src/hripcb_preprocessing/incremental.py \
  tests/test_preprocessing_runner_resume.py
git commit -m "feat: support resumable candidate batches"
```

### Task 4: Dedicated Member 5 Full-Search Script

**Files:**
- Create: `scripts/run_member5_full_search.py`
- Create: `tests/test_member5_full_search.py`

**Interfaces:**
- Consumes: `build_candidates`, `run_sweep(..., candidates=batch)`, and Task 3 lifecycle helpers.
- Produces: `run_search(dataset, output, config, *, batch_size=8, keep_variants=False, project_output=None, batch_runner=None) -> Path`.
- Produces: `build_summary(records: list[dict]) -> dict`.
- Produces CLI flags: `--dataset`, `--output`, `--config`, `--batch-size`, `--keep-variants`, `--project-output`.

- [ ] **Step 1: Write failing summary-selection tests**

Create `tests/test_member5_full_search.py` with an import helper and test:

```python
import importlib


def _search_module():
    return importlib.import_module("scripts.run_member5_full_search")


def _record(candidate_id, technique, score):
    return {
        "id": candidate_id,
        "module": "member5",
        "technique": technique,
        "split": "val",
        "evaluation_type": "ablation",
        "parameters": {},
        "metrics": {"map50_95": score},
    }


def test_summary_selects_winner_only_from_mandatory_combinations():
    records = [
        _record("original", "original", 0.5151),
        _record("tv", "tv", 0.90),
        _record("morph", "top_black_hat", 0.80),
        _record("combined_low", "tv_top_black_hat", 0.52),
        _record("combined_high", "tv_top_black_hat", 0.53),
    ]

    summary = _search_module().build_summary(records)

    assert summary["best_tv"]["id"] == "tv"
    assert summary["best_top_black_hat"]["id"] == "morph"
    assert summary["best_combined"]["id"] == "combined_high"
    assert summary["combined_improvement_vs_original"] == pytest.approx(0.0149)
```

- [ ] **Step 2: Write a failing interruption/resume/cleanup test**

Use a four-candidate miniature config and an injected fake batch runner:

```python
import json
import shutil
from pathlib import Path

import pytest


def test_search_persists_completed_batch_then_resumes_remaining(tmp_path):
    module = _search_module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "member5"
    config = {
        "module": "member5",
        "tv_weights": [0.02],
        "morphology_kernel_sizes": [5],
        "top_hat_amounts": [0.5],
        "black_hat_amounts": [0.5],
    }
    calls = []

    def fake_batch_runner(_dataset, batch_output, _config, *, candidates):
        calls.append([row["id"] for row in candidates])
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        (batch_output / "previews").mkdir(parents=True)
        records = []
        for candidate in candidates:
            preview = batch_output / "previews" / f"{candidate['id']}.jpg"
            preview.write_bytes(b"preview")
            records.append({
                **candidate,
                "split": "val",
                "evaluation_type": "ablation",
                "metrics": {"map50_95": 0.5},
                "preview": f"previews/{candidate['id']}.jpg",
            })
        (batch_output / "results.json").write_text(json.dumps(records))
        return batch_output

    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.run_search(
            dataset,
            output,
            config,
            batch_size=2,
            batch_runner=fake_batch_runner,
        )

    progress = json.loads((output / "progress.json").read_text())
    completed = {row["id"] for row in progress["records"]}
    assert completed == set(calls[0])
    assert not (output / "_batches" / "batch_001_002").exists()

    remaining_calls = []

    def successful_runner(_dataset, batch_output, _config, *, candidates):
        remaining_calls.append([row["id"] for row in candidates])
        (batch_output / "previews").mkdir(parents=True)
        records = []
        for candidate in candidates:
            preview = batch_output / "previews" / f"{candidate['id']}.jpg"
            preview.write_bytes(b"preview")
            records.append({
                **candidate,
                "split": "val",
                "evaluation_type": "ablation",
                "metrics": {"map50_95": 0.5},
                "preview": f"previews/{candidate['id']}.jpg",
            })
        (batch_output / "results.json").write_text(json.dumps(records))
        return batch_output

    module.run_search(
        dataset,
        output,
        config,
        batch_size=2,
        batch_runner=successful_runner,
    )

    assert all(completed.isdisjoint(batch) for batch in remaining_calls)
    assert len(json.loads((output / "results.json").read_text())) == 4
```

- [ ] **Step 3: Add failing tests for config mismatch and keep-variants**

Add two focused tests using the same fake runner:

```python
def test_search_rejects_resume_after_result_affecting_config_change(tmp_path):
    module = _search_module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "member5"
    first = {
        "module": "member5",
        "tv_weights": [0.01],
        "morphology_kernel_sizes": [5],
        "top_hat_amounts": [0.5],
        "black_hat_amounts": [0.5],
    }
    fingerprint = module.config_fingerprint(first, dataset)
    module.atomic_write_json(output / "progress.json", {
        "version": 1,
        "fingerprint": fingerprint,
        "records": [],
    })

    with pytest.raises(ValueError, match="different configuration"):
        module.run_search(dataset, output, {**first, "tv_weights": [0.02]})


def test_keep_variants_preserves_successful_batch_directories(tmp_path):
    module = _search_module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "member5"
    config = {
        "module": "member5",
        "tv_weights": [0.02],
        "morphology_kernel_sizes": [5],
        "top_hat_amounts": [0.5],
        "black_hat_amounts": [0.5],
    }

    def fake_batch_runner(_dataset, batch_output, _config, *, candidates):
        (batch_output / "previews").mkdir(parents=True)
        records = []
        for candidate in candidates:
            preview = batch_output / "previews" / f"{candidate['id']}.jpg"
            preview.write_bytes(b"preview")
            records.append({
                **candidate,
                "split": "val",
                "evaluation_type": "ablation",
                "metrics": {"map50_95": 0.5},
                "preview": f"previews/{candidate['id']}.jpg",
            })
        (batch_output / "results.json").write_text(json.dumps(records))
        return batch_output

    module.run_search(
        dataset,
        output,
        config,
        batch_size=2,
        keep_variants=True,
        batch_runner=fake_batch_runner,
    )

    assert any((output / "_batches").iterdir())


def test_progress_write_failure_keeps_completed_batch_files(tmp_path, monkeypatch):
    module = _search_module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "member5"
    config = {
        "module": "member5",
        "tv_weights": [0.02],
        "morphology_kernel_sizes": [5],
        "top_hat_amounts": [0.5],
        "black_hat_amounts": [0.5],
    }

    def fake_batch_runner(_dataset, batch_output, _config, *, candidates):
        (batch_output / "previews").mkdir(parents=True)
        records = []
        for candidate in candidates:
            preview = batch_output / "previews" / f"{candidate['id']}.jpg"
            preview.write_bytes(b"preview")
            records.append({
                **candidate,
                "split": "val",
                "evaluation_type": "ablation",
                "metrics": {"map50_95": 0.5},
                "preview": f"previews/{candidate['id']}.jpg",
            })
        (batch_output / "results.json").write_text(json.dumps(records))
        return batch_output

    def fail_progress_write(_path, _payload):
        raise OSError("simulated progress write failure")

    monkeypatch.setattr(module, "atomic_write_json", fail_progress_write)

    with pytest.raises(OSError, match="progress write failure"):
        module.run_search(
            dataset,
            output,
            config,
            batch_size=4,
            batch_runner=fake_batch_runner,
        )

    assert (output / "_batches" / "batch_001_004").exists()
```

These tests use only temporary files and never load the real dataset or YOLO.

- [ ] **Step 4: Run script tests and verify RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_member5_full_search.py
```

Expected: import failure because the script does not exist.

- [ ] **Step 5: Implement deterministic batching and summary helpers**

Create `scripts/run_member5_full_search.py`. The core helpers use stable candidate
order and original indices for batch directory names:

```python
def pending_batches(candidates, completed_ids, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indexed = [
        (index, candidate)
        for index, candidate in enumerate(candidates, start=1)
        if candidate["id"] not in completed_ids
    ]
    for offset in range(0, len(indexed), batch_size):
        chunk = indexed[offset:offset + batch_size]
        yield chunk[0][0], chunk[-1][0], [row for _, row in chunk]


def _best(records, technique):
    matches = [row for row in records if row.get("technique") == technique]
    if not matches:
        raise ValueError(f"No results found for technique: {technique}")
    return max(
        matches,
        key=lambda row: (float(row["metrics"]["map50_95"]), row["id"]),
    )


def build_summary(records):
    original = _best(records, "original")
    best_combined = _best(records, "tv_top_black_hat")
    ranked = sorted(
        [row for row in records if row.get("technique") == "tv_top_black_hat"],
        key=lambda row: (float(row["metrics"]["map50_95"]), row["id"]),
        reverse=True,
    )
    return {
        "primary_metric": "map50_95",
        "candidate_count": len(records),
        "combined_candidate_count": len(ranked),
        "best_tv": _best(records, "tv"),
        "best_top_black_hat": _best(records, "top_black_hat"),
        "best_combined": best_combined,
        "top_combined": ranked,
        "original_map50_95": float(original["metrics"]["map50_95"]),
        "combined_improvement_vs_original": (
            float(best_combined["metrics"]["map50_95"])
            - float(original["metrics"]["map50_95"])
        ),
    }
```

- [ ] **Step 6: Implement the resumable batch loop**

Implement `run_search` with this sequence:

```python
def run_search(
    dataset,
    output,
    config,
    *,
    batch_size=8,
    keep_variants=False,
    project_output=None,
    batch_runner=None,
):
    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if str(config.get("module")) != "member5":
        raise ValueError("Member 5 search requires module: member5")
    candidates = build_candidates("member5", config)
    fingerprint = config_fingerprint(config, dataset)
    progress_path = output / "progress.json"
    progress = load_progress(progress_path, fingerprint)
    records_by_id = {row["id"]: row for row in progress["records"]}
    batches_root = output / "_batches"
    runner = batch_runner or run_sweep

    for first, last, batch in pending_batches(
        candidates, set(records_by_id), batch_size
    ):
        batch_root = batches_root / f"batch_{first:03d}_{last:03d}"
        if batch_root.exists():
            remove_completed_batch(batch_root, batches_root)
        runner(dataset, batch_root, config, candidates=batch)
        batch_records = json.loads(
            (batch_root / "results.json").read_text(encoding="utf-8")
        )
        preview_root = output / "previews"
        preview_root.mkdir(parents=True, exist_ok=True)
        for record in batch_records:
            source = batch_root / record["preview"]
            destination = preview_root / f"{record['id']}.jpg"
            shutil.copy2(source, destination)
            record["preview"] = f"previews/{record['id']}.jpg"
            records_by_id[record["id"]] = record
        progress = {
            "version": 1,
            "fingerprint": fingerprint,
            "records": [
                records_by_id[row["id"]]
                for row in candidates
                if row["id"] in records_by_id
            ],
        }
        atomic_write_json(progress_path, progress)
        if not keep_variants:
            remove_completed_batch(batch_root, batches_root)

    records = [records_by_id[row["id"]] for row in candidates]
    atomic_write_json(output / "results.json", records)
    _write_flat_csv(output / "results.csv", records)
    atomic_write_json(output / "summary.json", build_summary(records))
    write_dashboard_html(
        output,
        records,
        title="Member 5 / TV + Top-hat & Black-hat Full Search",
        primary_metric="map50_95",
    )
    return output / "summary.json"
```

The final implementation must import `json`, `shutil`, `Path`, the Task 3 helpers,
`build_candidates`, `run_sweep`, `_write_flat_csv`, and `write_dashboard_html`.
Do not call project aggregation in this task; Task 5 adds it after the completed
result source is recognized.

- [ ] **Step 7: Add the CLI without running it**

Use these defaults:

```python
parser.add_argument("--dataset", type=Path, default=Path("HRIPCB_UPDATE"))
parser.add_argument("--output", type=Path, default=Path("runs/member5_full_search"))
parser.add_argument("--config", type=Path, default=Path("configs/member5_full_search.yaml"))
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--keep-variants", action="store_true")
parser.add_argument(
    "--project-output",
    type=Path,
    default=Path("runs/project_validation_comparison"),
)
```

Load the YAML through `load_config`, call `run_search`, and print the summary path
plus the best-combined ID and `mAP50-95`. Task 5 wires `project_output` into final
aggregation.

- [ ] **Step 8: Run script unit tests and existing runner tests**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_member5_full_search.py \
  tests/test_preprocessing_runner_resume.py
```

Expected: all tests pass without loading YOLO or processing HRIPCB images.

- [ ] **Step 9: Commit the dedicated search script**

```bash
git add scripts/run_member5_full_search.py tests/test_member5_full_search.py
git commit -m "feat: add resumable Member 5 full search"
```

### Task 5: Complete-Only Project Aggregation

**Files:**
- Modify: `scripts/build_project_dashboard.py`
- Modify: `scripts/run_member5_full_search.py`
- Modify: `tests/test_project_dashboard_sorting.py`
- Modify: `tests/test_member5_full_search.py`

**Interfaces:**
- Produces: project `MODULES = ("member1", ..., "member5")`.
- Produces: `aggregate_results(runs_root, output_root, *, member5_source=None) -> Path`.
- Consumes a completed Member 5 `results.json` only.
- Extends `run_search(..., project_output=Path(...))` to publish after finalization.

- [ ] **Step 1: Add failing aggregation-preservation test**

Add a test that starts with a tracked aggregate containing Member 1-4 and then
adds a complete Member 5 source:

```python
def test_aggregate_adds_member5_without_removing_existing_members(tmp_path):
    runs_root = tmp_path / "runs"
    output = runs_root / "project_validation_comparison"
    output.mkdir(parents=True)
    existing = [
        {
            "id": f"member{i}_combined",
            "module": f"member{i}",
            "technique": "gaussian_bbhe",
            "split": "val",
            "evaluation_type": "ablation",
            "metrics": {"map50_95": 0.5},
        }
        for i in range(1, 5)
    ]
    (output / "results.json").write_text(json.dumps(existing))
    member5 = runs_root / "member5_full_search"
    member5.mkdir()
    (member5 / "results.json").write_text(json.dumps([{
        "id": "member5_combined",
        "module": "member5",
        "technique": "tv_top_black_hat",
        "split": "val",
        "evaluation_type": "ablation",
        "metrics": {"map50_95": 0.52},
    }]))

    aggregate_results(runs_root, output)
    records = json.loads((output / "results.json").read_text())

    assert {row["module"] for row in records} == {
        "member1", "member2", "member3", "member4", "member5"
    }
```

- [ ] **Step 2: Run the aggregation test and verify RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_project_dashboard_sorting.py \
  -k member5
```

Expected: failure because `MODULES` and source discovery stop at Member 4.

- [ ] **Step 3: Recognize the completed Member 5 source**

Change `MODULES` in `build_project_dashboard.py` to:

```python
MODULES = ("member1", "member2", "member3", "member4", "member5")
```

Replace inline source selection with this helper and use it in the source list:

```python
def _module_source(
    runs_root: Path,
    module: str,
    member5_source: Path | None = None,
) -> Path:
    if module == "member2" and (runs_root / "member2_full_search" / "results.json").is_file():
        return runs_root / "member2_full_search"
    if module == "member5":
        return Path(member5_source) if member5_source else runs_root / "member5_full_search"
    return runs_root / f"{module}_validation_sweep"


def aggregate_results(
    runs_root: Path,
    output_root: Path,
    *,
    member5_source: Path | None = None,
) -> Path:
    sources = [
        (_module_source(runs_root, module, member5_source), module)
        for module in MODULES
    ]
```

Keep the existing aggregation statements before and after `sources`; only the
signature and source-list construction change.

Update the module-level docstring from Member 1-4 to Member 1-5.

- [ ] **Step 4: Publish project results only after all records finalize**

At the end of `run_search`, after `results.json`, CSV, summary, and HTML are
written, add:

```python
if project_output is not None:
    aggregate_results(
        output.parent,
        Path(project_output),
        member5_source=output,
    )
```

Import `aggregate_results` from `scripts.build_project_dashboard`. Add a script
test with an injected or monkeypatched aggregator and assert it is not called
when the batch runner raises, then is called exactly once after a successful
complete run.

- [ ] **Step 5: Run aggregation and search tests**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_project_dashboard_sorting.py \
  tests/test_member5_full_search.py
```

Expected: all tests pass; an interrupted search leaves the project aggregate
unchanged.

- [ ] **Step 6: Commit complete-only aggregation**

```bash
git add scripts/build_project_dashboard.py \
  scripts/run_member5_full_search.py \
  tests/test_project_dashboard_sorting.py \
  tests/test_member5_full_search.py
git commit -m "feat: publish completed Member 5 results"
```

### Task 6: Five-Member Dashboard, Analysis, and Inference UI

**Files:**
- Modify: `src/hripcb_dashboard/filtering.py`
- Modify: `src/hripcb_dashboard/analysis.py`
- Modify: `src/hripcb_dashboard/dashboard.py`
- Modify: `scripts/streamlit_dashboard.py`
- Modify: `tests/test_dashboard_filtering.py`
- Modify: `tests/test_dashboard_extra_effort.py`
- Modify: `tests/test_experiment_dashboard.py`

**Interfaces:**
- Produces label mapping for `tv`, `top_black_hat`, and `tv_top_black_hat`.
- Produces `MEMBER_TECHNIQUES["member5"] == ("tv", "top_black_hat")`.
- Produces combined recognition for `tv_top_black_hat` in Python and static JS.
- Consumes Member 5 result records for generic image/video inference through `apply_candidate`.

- [ ] **Step 1: Add failing filtering and shared-control tests**

Extend the test fixtures with Member 5 and assert:

```python
def test_member5_combined_is_ranked_and_shared_original_is_collapsed():
    records = [
        {
            "id": "original",
            "model_id": "baseline",
            "module": f"member{i}",
            "technique": "original",
            "split": "val",
        }
        for i in range(1, 6)
    ]
    records.append({
        "id": "member5_combined",
        "model_id": "baseline",
        "module": "member5",
        "technique": "tv_top_black_hat",
        "split": "val",
        "evaluation_type": "ablation",
        "metrics": {"map50_95": 0.53},
    })

    assert is_combined_record(records[-1]) is True
    assert best_by_module(records)[0]["id"] == "member5_combined"
    control = collapse_shared_baseline(records)[0]
    assert control["shared_control_modules"] == [
        "member1", "member2", "member3", "member4", "member5"
    ]
```

- [ ] **Step 2: Add failing analysis and static-dashboard tests**

Update the existing four-member fixture in
`tests/test_dashboard_extra_effort.py` to generate five combined records, using
`tv_top_black_hat` for Member 5. Assert six display rows: one shared original and
five combined winners. In `tests/test_experiment_dashboard.py`, add a Member 5
record and assert the generated HTML contains `tv_top_black_hat` and all four
parameter names.

- [ ] **Step 3: Run dashboard tests and verify RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_dashboard_filtering.py \
  tests/test_dashboard_extra_effort.py \
  tests/test_experiment_dashboard.py
```

Expected: failures because Python and JavaScript constants still recognize four
members and four combined techniques.

- [ ] **Step 4: Extend filtering and analysis constants**

In `filtering.py`, add `tv_top_black_hat` to `COMBINED_TECHNIQUES` and change:

```python
MEMBER_MODULES = ("member1", "member2", "member3", "member4", "member5")
```

In `analysis.py`, add:

```python
"tv": "Total Variation",
"top_black_hat": "Top-hat + Black-hat",
"tv_top_black_hat": "TV + Top-hat + Black-hat",
```

and:

```python
"member5": ("tv", "top_black_hat"),
```

Update the analysis docstring from five displayed rows/four modules to six
displayed rows/five modules.

- [ ] **Step 5: Extend static-dashboard recognition**

In `dashboard.py`, add `tv_top_black_hat` to the JavaScript `combinedTechniques`
set and `member5` to the `isSharedOriginal` list. No layout rewrite is required;
the existing filters and parameter-key discovery are data-driven.

- [ ] **Step 6: Update Streamlit wording and Member 5 callout**

Change the comparison caption to “five member combined techniques” and the app
caption to “Member 1–5 preprocessing experiments.” After the Member 2 preset
callout, add:

```python
member5 = next(
    (row for row in best_by_module(records) if row.get("module") == "member5"),
    None,
)
if member5:
    parameters = member5.get("parameters", {})
    st.info(
        "Member 5 final assignment preset (TV → Top-hat & Black-hat): "
        f"weight={parameters.get('tv_weight')}, "
        f"kernel={parameters.get('morphology_kernel_size')}, "
        f"top amount={parameters.get('top_hat_amount')}, "
        f"black amount={parameters.get('black_hat_amount')}; "
        f"validation mAP50-95={_metric_value(member5, 'map50_95'):.4f}."
    )
```

The existing exact-parameter expander, `_candidate_from_record`, image inference,
and video inference should remain generic; Task 2 makes their Member 5 dispatch
work without duplicate preprocessing code.

- [ ] **Step 7: Run dashboard and inference regressions**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_dashboard_filtering.py \
  tests/test_dashboard_extra_effort.py \
  tests/test_experiment_dashboard.py \
  tests/test_streamlit_inference.py \
  tests/test_streamlit_deployment.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit five-member UI integration**

```bash
git add src/hripcb_dashboard/filtering.py \
  src/hripcb_dashboard/analysis.py \
  src/hripcb_dashboard/dashboard.py \
  scripts/streamlit_dashboard.py \
  tests/test_dashboard_filtering.py \
  tests/test_dashboard_extra_effort.py \
  tests/test_experiment_dashboard.py
git commit -m "feat: integrate Member 5 into dashboards"
```

### Task 7: Usage Documentation and Safe Final Verification

**Files:**
- Modify: `README.md`
- Modify: `scripts/run_preprocessing_sweep.py`
- Test: complete repository test suite.

**Interfaces:**
- Documents the user-run command and output paths.
- Does not run the 52-candidate sweep.

- [ ] **Step 1: Update the member-method table and manual command**

Add Member 5 to the validation-sweep table:

```markdown
| member5 | TV weight `0.01, 0.02, 0.05`; elliptical Top-hat/Black-hat kernel `5, 9, 15`; independent amounts `0.5, 1.0` |
```

Document the dedicated command separately from `run_all_validation_sweeps.py`:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_member5_full_search.py
```

State that it evaluates 52 validation candidates in resumable batches, removes
successful batch datasets by default, accepts `--keep-variants`, and updates the
project result JSON only after completion. State explicitly that the user, not
the implementation verification workflow, runs this command.

Change the README introduction from “four member modules” to “five member
modules” and change the Analysis & reports description from “four member
modules” to “five member modules.” Change the generic CLI docstring in
`scripts/run_preprocessing_sweep.py` to:

```python
"""Run one generic member preprocessing sweep."""
```

- [ ] **Step 2: Verify candidate construction without preprocessing images**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python - <<'PY'
from pathlib import Path
import yaml
from hripcb_preprocessing.candidates import build_candidates

config = yaml.safe_load(Path("configs/member5_full_search.yaml").read_text())
rows = build_candidates("member5", config)
combined = [row for row in rows if row["technique"] == "tv_top_black_hat"]
assert len(rows) == 52
assert len(combined) == 36
assert len({row["id"] for row in rows}) == 52
print("Member 5 grid: 52 total / 36 combined")
PY
```

Expected: prints `Member 5 grid: 52 total / 36 combined` without loading YOLO.

- [ ] **Step 3: Run the complete automated test suite**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q
```

Expected: zero failures. Record the exact pass count in the handoff.

- [ ] **Step 4: Run import, syntax, and whitespace verification**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m compileall -q src scripts tests
git diff --check
git status --short
```

Expected: compileall and diff check exit zero. The status contains only intended
Member 5 changes, if any remain uncommitted.

- [ ] **Step 5: Confirm the full sweep was not launched**

Run:

```bash
test ! -f runs/member5_full_search/results.json
```

Expected in a fresh implementation checkout: exit zero. If the user already ran
the script independently, do not delete or overwrite their results; report that
the output exists and inspect it read-only.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md scripts/run_preprocessing_sweep.py
git commit -m "docs: explain Member 5 full search"
```

## Execution Handoff

After implementation and automated verification, give the user this command but
do not execute it:

```bash
cd /Users/ng/Documents/GitHub/Image_Processing
PYTHONPATH=src ./.venv/bin/python scripts/run_member5_full_search.py
```

Explain that rerunning the same command resumes completed batches. After all 52
candidates finish, the user can review:

```text
runs/member5_full_search/summary.json
runs/member5_full_search/dashboard.html
runs/project_validation_comparison/results.json
```

Do not claim a Member 5 winner or improvement until those completed files exist
and have been inspected.
