# Member 3 Formal Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the agreed 16-condition, validation-only Member 3 preprocessing study and show its isolated results in the Streamlit dashboard.

**Architecture:** Keep the legacy noise experiment unchanged. Add a focused module for formal condition definitions, Y-channel preprocessing, quality metrics, dataset preparation and normalized records. A dedicated CLI evaluates only validation data; the dashboard selects formal presets and reads only formal validation results.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, Ultralytics YOLOv8s, Streamlit, pytest.

## Global Constraints

- Use `runs/baseline/weights/best.pt`, split `val`, 138 images, `imgsz=1024`, `conf=0.25`, `iou=0.70`, `workers=0`, and `device=auto`.
- Rank only by descending `mAP50-95`; do not read, create, or rank test rows.
- Use exactly 1 Original, 3 Bilateral, 3 AGCWD plus gamma, and 9 combined conditions.
- Use Bilateral `(5,25,25)`, `(7,50,50)`, `(9,75,75)`; gamma `0.8`, `1.0`, `1.2`; fixed AGCWD alpha `0.75`.
- Preserve Cr/Cb and transform only Y in YCrCb.
- Keep legacy `runs/member3/` artefacts preliminary and unchanged; write formal results to `runs/member3_formal/`.

---

### Task 1: Formal condition definitions and image metrics

**Files:**
- Create: `src/hripcb_baseline/member3_formal.py`
- Modify: `requirements.txt`
- Test: `tests/test_member3_formal.py`

**Interfaces:**
- `FormalCondition(identifier: str, technique: str, bilateral: BilateralConfig | None, gamma: float)`.
- `build_formal_conditions() -> list[FormalCondition]`.
- `apply_formal_condition(image: np.ndarray, condition: FormalCondition) -> np.ndarray`.
- `measure_image_quality(original: np.ndarray, processed: np.ndarray) -> ImageQuality`.

- [ ] **Step 1: Write the failing condition-matrix test.**

```python
def test_build_formal_conditions_contains_the_agreed_sixteen_variants():
    conditions = build_formal_conditions()
    assert len(conditions) == 16
    assert [item.technique for item in conditions].count("original") == 1
    assert [item.technique for item in conditions].count("bilateral") == 3
    assert [item.technique for item in conditions].count("agcwd_gamma") == 3
    assert [item.technique for item in conditions].count("combined") == 9
```

- [ ] **Step 2: Run the test red.**

Run: `python3 -m pytest -q tests/test_member3_formal.py -k sixteen`

Expected: import error because `member3_formal` does not exist.

- [ ] **Step 3: Implement the immutable condition matrix.**

```python
FORMAL_BILATERAL_PRESETS = (
    BilateralConfig(5, 25.0, 25.0),
    BilateralConfig(7, 50.0, 50.0),
    BilateralConfig(9, 75.0, 75.0),
)
FORMAL_GAMMAS = (0.8, 1.0, 1.2)
AGCWD_ALPHA = 0.75
```

Implement global gamma after AGCWD with `round(255 * (Y / 255) ** gamma)`, clipped to `uint8`. Compute RGB PSNR with OpenCV and mean per-channel Gaussian-window SSIM with OpenCV/NumPy.

- [ ] **Step 4: Write and run processing tests.**

Test original copies its input, every preset preserves RGB shape/dtype, gamma `1.0` is identity, and identical images produce `PSNR=inf`, `SSIM=1.0`.

Run: `python3 -m pytest -q tests/test_member3_formal.py`

Expected: all condition and quality tests pass.

### Task 2: Validation dataset preparation and normalized records

**Files:**
- Modify: `src/hripcb_baseline/member3_formal.py`
- Test: `tests/test_member3_formal.py`

**Interfaces:**
- `prepare_formal_validation_dataset(dataset_root: Path, output_root: Path, condition: FormalCondition) -> PreparedCondition`.
- `FormalResultRecord(...).as_dict() -> dict[str, object]`.
- `rank_formal_results(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]`.

- [ ] **Step 1: Write the failing preparation test.**

```python
def test_prepare_formal_validation_dataset_copies_labels_and_aggregates_metrics(tmp_path):
    source = make_validation_dataset(tmp_path / "source", image_count=2)
    prepared = prepare_formal_validation_dataset(
        source, tmp_path / "output", build_formal_conditions()[1]
    )
    assert (prepared.dataset_root / "val/images/board_0.jpg").is_file()
    assert (prepared.dataset_root / "val/labels/board_0.txt").is_file()
    assert prepared.processing_time_ms >= 0.0
    assert 0.0 <= prepared.ssim <= 1.0
```

- [ ] **Step 2: Run the test red.**

Run: `python3 -m pytest -q tests/test_member3_formal.py -k preparation`

Expected: import error because `prepare_formal_validation_dataset` does not exist.

- [ ] **Step 3: Implement validation-only dataset preparation.**

Create `<output>/<condition-id>/val/images` and `labels`; require every source image to have a matching label; copy/process images and copy labels. Aggregate mean processing milliseconds, PSNR and SSIM. Original copies image files and returns `0.0`, `inf`, and `1.0`. Write a local YOLO YAML for this condition.

- [ ] **Step 4: Add and run record-schema tests.**

Require every record to contain model/checkpoint/member/technique/training preprocessing/evaluation preprocessing/dataset split/settings/parameters/detection metrics/quality metrics/time. Set `dataset_split="val"`, `member="Member 3"`, and `primary_metric="mAP50-95"`. Verify ranking breaks equal mAP ties by condition identifier.

Run: `python3 -m pytest -q tests/test_member3_formal.py`

Expected: all formal module tests pass.

### Task 3: Formal validation CLI

**Files:**
- Create: `scripts/run_member3_formal.py`
- Create: `tests/test_member3_formal_cli.py`

**Interfaces:**
- Command: `python3 scripts/run_member3_formal.py --dataset-root HRIPCB_UPDATE`.
- Outputs: `runs/member3_formal/comparison.csv`, `metrics.json`, `summary.json`, generated validation data, and plots.

- [ ] **Step 1: Write failing CLI contract tests.**

```python
def test_formal_cli_rejects_wrong_validation_image_count(tmp_path):
    result = run_cli("--dataset-root", str(make_dataset(tmp_path, val_images=2)))
    assert result.returncode == 2
    assert "expected 138 validation images" in result.stderr

def test_formal_cli_defaults_are_the_shared_contract():
    args = parse_args(["--dataset-root", "HRIPCB_UPDATE"])
    assert args.weights == Path("runs/baseline/weights/best.pt")
    assert args.output == Path("runs/member3_formal")
```

- [ ] **Step 2: Run the test red.**

Run: `python3 -m pytest -q tests/test_member3_formal_cli.py`

Expected: import error because the formal CLI does not exist.

- [ ] **Step 3: Implement the fixed-contract runner.**

Accept `--dataset-root`, optional `--weights`, optional `--output`, and `--device`. Validate checkpoint and `val/images`/`val/labels`, reject any count other than 138, resolve MPS/CPU, prepare/evaluate all 16 conditions with `YOLO.val(split="val", imgsz=1024, conf=0.25, iou=0.70, workers=0)`, then write sorted CSV/JSON and a summary containing the best condition identifier.

- [ ] **Step 4: Add a fake-evaluator output test and run all CLI tests.**

Verify CSV sorts `map50_95` descending and `summary.json` selects the first row.

Run: `python3 -m pytest -q tests/test_member3_formal.py tests/test_member3_formal_cli.py`

Expected: pass without a real dataset or YOLO inference.

### Task 4: Formal dashboard controls and results

**Files:**
- Modify: `src/hripcb_baseline/member3_demo.py`
- Modify: `scripts/member3_demo.py`
- Modify: `tests/test_member3_demo.py`

**Interfaces:**
- The dashboard selects Original, Bilateral Filtering, AGCWD plus gamma, or Bilateral plus AGCWD plus gamma and their applicable presets.
- It reads only `runs/member3_formal/comparison.csv` records where `dataset_split == "val"`.

- [ ] **Step 1: Write failing summary-filter tests.**

```python
def test_formal_summary_filter_matches_exact_condition_identifier():
    rows = [{"condition_id": "combined_d7_c50_s50_g1", "dataset_split": "val"}]
    assert filter_formal_summary_rows(rows, "combined_d7_c50_s50_g1") == rows
    assert filter_formal_summary_rows(rows, "missing") == []
```

- [ ] **Step 2: Run the test red, then implement pure dashboard helpers.**

Run: `python3 -m pytest -q tests/test_member3_demo.py -k formal_summary`

Expected: import error before implementation; pass afterward.

- [ ] **Step 3: Replace noise controls with formal preset controls.**

Build the selected `FormalCondition`, apply it to uploaded RGB images, and caption each image with the exact technique/preset. Prefer formal CSV; show an actionable validation-run message when missing. Default to the matching one-row result and provide a `Show all formal experiments` control. Render all requested metadata and metrics; do not load legacy test rows.

- [ ] **Step 4: Run dashboard tests and compilation.**

Run: `python3 -m pytest -q tests/test_member3_demo.py && python3 -m py_compile scripts/member3_demo.py`

Expected: all dashboard tests pass and script compilation exits zero.

### Task 5: Documentation and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the exact formal command.**

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_member3_formal.py \
  --dataset-root /path/to/HRIPCB_UPDATE \
  --weights runs/baseline/weights/best.pt \
  --output runs/member3_formal
```

State that this is validation-only Member 3 tuning, writes 16 results, selects by mAP50-95, and must not support a final test claim.

- [ ] **Step 2: Run focused and full verification.**

Run: `python3 -m pytest -q tests/test_member3.py tests/test_member3_runner.py tests/test_member3_experiment.py tests/test_member3_formal.py tests/test_member3_formal_cli.py tests/test_member3_demo.py`

Run: `python3 -m pytest -q`

Expected: report the existing missing-external-dataset test separately if it remains; do not rewrite dataset paths.

- [ ] **Step 3: Inspect the final diff.**

Run: `git diff --check && git status --short`

Expected: no whitespace errors in changed source/test/documentation files and no modification to legacy result artefacts.

## Plan Self-Review

- Tasks 1-3 cover every formal preprocessing variant, fixed validation evaluation, metrics, ranking, and output record.
- Task 4 keeps the UI isolated to formal validation results and removes misleading noise controls.
- Task 5 documents the handoff and verifies all Member 3 behavior.
