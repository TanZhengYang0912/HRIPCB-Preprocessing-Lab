# Canonical Bilateral + AGCWD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mandatory two-method Member 3 search using canonical HSV-V AGCWD, positive blending, and both Bilateral/AGCWD orders.

**Architecture:** Keep filter math in `filters.py`, experiment serialization/order orchestration in `candidates.py`, and reuse the generic sweep runner. Keep the `bilateral_agcwd` technique name for dashboard compatibility; explicit alpha, blend, and order parameters make candidates reproducible.

**Tech Stack:** Python, NumPy, OpenCV, PyYAML, pytest, and the existing Ultralytics evaluation runner.

## Global Constraints

- Both methods execute for every combined candidate; `agcwd_blend >= 0.05`.
- Select only on `val`; never tune on `test`.
- Primary gate: `mAP50-95 > 0.5151173658`; stretch target: `0.5235315781`.
- Preserve checkpoint, `imgsz=1024`, `conf=0.25`, `iou=0.70`, and `workers=0`.
- Do not change Members 1, 2, or 4.
- Every production change requires an observed failing test first.

## File Map

- `src/hripcb_preprocessing/filters.py`: canonical AGCWD and image blending.
- `src/hripcb_preprocessing/candidates.py`: Member 3 grid and order dispatch.
- `configs/member3_sweep.yaml`: alpha/blend/order search space.
- `tests/test_preprocessing_modules.py`: behavior, validation, grid, and staging tests.
- `README.md`: new search description without an unmeasured score claim.

### Task 1: Canonical HSV-V AGCWD and Blend Primitive

**Files:**
- Modify: `tests/test_preprocessing_modules.py`
- Modify: `src/hripcb_preprocessing/filters.py:40-63`

**Interfaces:**
- Produces `apply_agcwd(image: np.ndarray, alpha: float = 0.5) -> np.ndarray`.
- Produces `blend_images(base: np.ndarray, enhanced: np.ndarray, strength: float) -> np.ndarray`.

- [ ] **Step 1: Write failing filter tests**

Add these imports and tests:

```python
from hripcb_preprocessing.filters import apply_agcwd, blend_images

def test_agcwd_rejects_alpha_outside_paper_range():
    for alpha in (0.0, -0.1, 1.01):
        with pytest.raises(ValueError, match="alpha"):
            apply_agcwd(_image(), alpha=alpha)

def test_agcwd_preserves_hue_and_saturation_with_rounding_tolerance():
    hsv = np.zeros((16, 16, 3), dtype=np.uint8)
    hsv[..., 0], hsv[..., 1] = 42, 180
    hsv[..., 2] = np.arange(16, dtype=np.uint8)[None, :] * 12 + 40
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    actual = cv2.cvtColor(apply_agcwd(image, alpha=0.5), cv2.COLOR_BGR2HSV)
    assert np.max(np.abs(actual[..., 0].astype(int) - hsv[..., 0].astype(int))) <= 2
    assert np.max(np.abs(actual[..., 1].astype(int) - hsv[..., 1].astype(int))) <= 3

@pytest.mark.parametrize("value", [0, 127, 255])
def test_agcwd_leaves_constant_images_stable(value):
    image = np.full((12, 12, 3), value, dtype=np.uint8)
    assert np.array_equal(apply_agcwd(image, alpha=0.5), image)

def test_blend_images_has_exact_endpoints():
    base = np.full((4, 5, 3), 20, dtype=np.uint8)
    enhanced = np.full((4, 5, 3), 220, dtype=np.uint8)
    assert np.array_equal(blend_images(base, enhanced, 0.0), base)
    assert np.array_equal(blend_images(base, enhanced, 1.0), enhanced)
    assert np.all(blend_images(base, enhanced, 0.25) == 70)
```

Also reject strength outside `[0, 1]`, mismatched shapes, and non-`uint8` inputs.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_preprocessing_modules.py -k 'agcwd or blend_images'`.
Expected: collection/API failure because the new helper/signature is absent.

- [ ] **Step 3: Implement the smallest canonical filter**

`apply_agcwd` validates `0 < alpha <= 1`, converts BGR to HSV, computes the V-channel PDF, the paper's weighted PDF using `pdf_min`, `pdf_max`, and `alpha`, then uses `gamma = 1 - weighted_cdf` to create a LUT. Preserve constant images as a degenerate no-op. `blend_images` validates both images and returns `cv2.addWeighted(base, 1-strength, enhanced, strength, 0)` with exact endpoint copies.

- [ ] **Step 4: Verify GREEN**

Run `pytest -q tests/test_preprocessing_modules.py -k 'agcwd or blend_images'`, then `pytest -q tests/test_preprocessing_modules.py`. Expected: all pass.

- [ ] **Step 5: Commit**

Run `git add src/hripcb_preprocessing/filters.py tests/test_preprocessing_modules.py && git commit -m "feat: implement canonical AGCWD blending"`.

### Task 2: Positive-Blend Grid and Two Orders

**Files:**
- Modify: `tests/test_preprocessing_modules.py`
- Modify: `src/hripcb_preprocessing/candidates.py:108-129,188-199`

**Interfaces:**
- Consumes `apply_agcwd(image, alpha)` and `blend_images(base, enhanced, strength)`.
- Produces parameters `agcwd_alpha`, `agcwd_blend`, `agcwd_order`.
- Orders: `bilateral_then_agcwd`, `agcwd_then_bilateral`.

- [ ] **Step 1: Write failing grid and staging tests**

Use this fixture:

```python
MEMBER3_CONFIG = {
    "bilateral_presets": [
        {"id": "b05", "diameter": 5, "sigma_color": 25, "sigma_space": 25},
    ],
    "agcwd_alphas": [0.25, 0.5],
    "agcwd_blends": [0.05, 0.2],
    "agcwd_orders": ["bilateral_then_agcwd", "agcwd_then_bilateral"],
}
```

Assert 12 candidates: one original, one Bilateral, two AGCWD controls, and eight combined. Assert every combined blend is at least `0.05`, both orders exist, and candidate IDs are unique. For each order, manually stage the same operations and require exact pixel equality with `apply_candidate`. Mutated candidates with blend `0.0` or order `sideways` must raise clear `ValueError`s.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_preprocessing_modules.py -k 'member3'`. Expected: failure because the builder still requires `agcwd_gammas`.

- [ ] **Step 3: Implement grid and dispatch**

Read `agcwd_alphas`, `agcwd_blends`, and `agcwd_orders`. Generate the Cartesian product of Bilateral preset × alpha × positive blend × order, keeping technique `bilateral_agcwd`. IDs include `a<alpha>`, `bl<blend>`, and `b2a`/`a2b`.

Dispatch semantics, after reading the Bilateral fields into local variables
`diameter`, `sigma_color`, and `sigma_space`:

```python
if order == "bilateral_then_agcwd":
    bilateral = apply_bilateral_filter(image, diameter, sigma_color, sigma_space)
    return blend_images(bilateral, apply_agcwd(bilateral, alpha), blend)
if order == "agcwd_then_bilateral":
    softened = blend_images(image, apply_agcwd(image, alpha), blend)
    return apply_bilateral_filter(softened, diameter, sigma_color, sigma_space)
raise ValueError(f"Unsupported AGCWD order: {order}")
```

Reject combined blend below `0.05`. AGCWD-only calls `apply_agcwd(image, agcwd_alpha)`.

- [ ] **Step 4: Verify GREEN**

Run `pytest -q tests/test_preprocessing_modules.py -k 'member3'`, then the whole file. Expected: all pass.

- [ ] **Step 5: Commit**

Run `git add src/hripcb_preprocessing/candidates.py tests/test_preprocessing_modules.py && git commit -m "feat: search blended Member 3 orders"`.

### Task 3: Repository Search Config and Documentation

**Files:**
- Modify: `configs/member3_sweep.yaml:18-31`
- Modify: `README.md:156-165,203-213`
- Modify: `tests/test_preprocessing_modules.py`

**Interfaces:**
- Produces exactly 128 configured candidates: 1 original, 3 Bilateral, 4 AGCWD, 120 combined.

- [ ] **Step 1: Write failing repository-config test**

```python
def test_member3_repository_config_builds_120_mandatory_combinations():
    import yaml
    from pathlib import Path
    config = yaml.safe_load(Path("configs/member3_sweep.yaml").read_text())
    candidates = build_candidates("member3", config)
    combined = [c for c in candidates if c["technique"] == "bilateral_agcwd"]
    assert len(candidates) == 128
    assert len(combined) == 120
    assert min(c["parameters"]["agcwd_blend"] for c in combined) == 0.05
```

- [ ] **Step 2: Verify RED**

Run the single test. Expected: failure because the repository config still contains `agcwd_gammas`.

- [ ] **Step 3: Update config and README**

Use:

```yaml
agcwd_alphas: [0.25, 0.5, 0.75, 1.0]
agcwd_blends: [0.05, 0.1, 0.15, 0.2, 0.3]
agcwd_orders: [bilateral_then_agcwd, agcwd_then_bilateral]
```

Document canonical HSV-V AGCWD, positive blending, two orders, and that no improved score exists before the sweep runs.

- [ ] **Step 4: Verify GREEN and regressions**

Run the config test, `pytest -q`, `python3 -m compileall -q src scripts tests`, and `git diff --check`. Expected: zero failures/errors.

- [ ] **Step 5: Check dataset and run the sweep when possible**

Run `python3 scripts/validate_dataset.py --root HRIPCB_UPDATE`. If all splits exist, run:

```bash
python3 scripts/run_preprocessing_sweep.py \
  --dataset HRIPCB_UPDATE \
  --config configs/member3_sweep.yaml \
  --output runs/member3_canonical_agcwd_validation
```

If splits are absent, preserve the exact failure and do not claim a new mAP.

- [ ] **Step 6: Commit**

Run `git add configs/member3_sweep.yaml README.md tests/test_preprocessing_modules.py && git commit -m "docs: configure canonical Member 3 sweep"`.

### Task 4: Independent Final Review

**Files:** Review only all changes after commit `572deb1`.

- [ ] **Step 1: Dispatch `luna_worker`**

Check equation fidelity, both order semantics, mandatory positive blend, ID uniqueness, stale configuration behavior, val/test leakage, and sensitive/absolute path exposure.

- [ ] **Step 2: Resolve confirmed issues through RED-GREEN**

For each confirmed problem: add one failing regression test, observe RED, apply the smallest fix, then observe GREEN.

- [ ] **Step 3: Run fresh final verification**

Run `pytest -q`, `python3 -m compileall -q src scripts tests`, `git diff --check`, and `git status --short`. Read every exit code before making a completion claim.
