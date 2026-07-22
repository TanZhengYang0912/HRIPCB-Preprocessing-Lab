# Member 1 Gaussian Filtering and BBHE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible image-only Member 1 module that creates controlled noisy/low-contrast HRIPCB test inputs, applies Gaussian Filtering and BBHE, and produces a visual comparison plus batch image-quality summaries.

**Architecture:** Keep pure image-processing functions in a small `src/hripcb_member1` package. A CLI will orchestrate deterministic degradation, processing, batch output, metrics, timing, and an HTML comparison page. The CLI will not import or load YOLO, `best.pt`, or any detector code.

**Tech Stack:** Python 3.13, OpenCV, NumPy, scikit-image, Pillow, PyYAML, pytest, static HTML.

## Global Constraints

- Use only `HRIPCB_UPDATE/test/images` and preserve the 70-image supplied test split.
- Use luminance-only processing through OpenCV YCrCb; preserve chroma channels.
- Use Gaussian noise `sigma=15,30,50` and contrast `alpha=0.75,0.50,0.25`.
- Use `sigma=30` and `alpha=0.50` for the representative visual page.
- Use Gaussian Filtering kernel `(5,5)` and `sigmaX=1.0`.
- Process combined degradation as noise then contrast, and combined restoration as Gaussian Filtering then BBHE.
- Use a deterministic global seed of `42` plus stable per-image/variant seed derivation.
- Do not load, modify, retrain, or evaluate `runs/baseline/weights/best.pt` in this phase.
- Do not modify any source image or label, and do not overwrite `runs/baseline/` or `runs/evaluation/`.
- Keep the existing baseline tests passing.

---

### Task 1: Add image-processing dependencies and pure-function package

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Create: `src/hripcb_member1/__init__.py`
- Create: `src/hripcb_member1/degradation.py`
- Create: `src/hripcb_member1/filters.py`
- Create: `tests/test_member1_processing.py`

**Interfaces:**
- `add_luminance_gaussian_noise(image: np.ndarray, sigma: float, seed: int) -> np.ndarray`
- `reduce_luminance_contrast(image: np.ndarray, alpha: float) -> np.ndarray`
- `apply_gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma_x: float = 1.0) -> np.ndarray`
- `apply_bbhe(image: np.ndarray) -> np.ndarray`
- All functions accept BGR `uint8` images and return same-shape BGR `uint8` images.

- [x] **Step 1: Write failing tests for degradation and filtering**

```python
def test_noise_is_deterministic_and_bounded(sample_image):
    first = add_luminance_gaussian_noise(sample_image, sigma=30, seed=42)
    second = add_luminance_gaussian_noise(sample_image, sigma=30, seed=42)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    assert int(first.min()) >= 0
    assert int(first.max()) <= 255


def test_contrast_reduction_moves_luminance_towards_midpoint(sample_image):
    result = reduce_luminance_contrast(sample_image, alpha=0.5)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8


def test_gaussian_filter_returns_same_shape_and_dtype(sample_image):
    result = apply_gaussian_filter(sample_image)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8


def test_bbhe_handles_constant_image_without_invalid_values():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    result = apply_bbhe(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.isfinite(result).all()
```

- [x] **Step 2: Run the focused tests and verify failure**

Run: `python3 -m pytest tests/test_member1_processing.py -q`

Expected: FAIL because the new package functions do not exist.

- [x] **Step 3: Add exact dependency declarations**

Add these runtime requirements while retaining the existing baseline requirements:

```text
numpy>=2,<3
opencv-python>=4.10,<6
scikit-image>=0.24,<1
```

Update `pyproject.toml` project dependencies to match the runtime requirements used by this package.

- [x] **Step 4: Implement deterministic luminance degradation**

```python
def _to_ycrcb(image: np.ndarray) -> np.ndarray:
    _validate_bgr_uint8(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


def add_luminance_gaussian_noise(image, sigma, seed):
    ycrcb = _to_ycrcb(image)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=ycrcb[..., 0].shape)
    ycrcb[..., 0] = np.clip(ycrcb[..., 0].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def reduce_luminance_contrast(image, alpha):
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    ycrcb = _to_ycrcb(image)
    y = ycrcb[..., 0].astype(np.float32)
    ycrcb[..., 0] = np.clip(128.0 + alpha * (y - 128.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
```

Use a shared input validator that rejects non-3-channel or non-`uint8` images, non-positive sigma, and invalid alpha values.

- [x] **Step 5: Implement Gaussian Filtering and BBHE**

```python
def apply_gaussian_filter(image, kernel_size=5, sigma_x=1.0):
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma_x <= 0:
        raise ValueError("sigma_x must be positive")
    _validate_bgr_uint8(image)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)


def apply_bbhe(image):
    ycrcb = _to_ycrcb(image)
    y = ycrcb[..., 0]
    mean_value = int(np.mean(y))
    output = np.empty_like(y)
    for low, high in ((0, mean_value), (mean_value + 1, 255)):
        mask = (y >= low) & (y <= high)
        if not np.any(mask) or low == high:
            output[mask] = y[mask]
            continue
        hist = np.bincount(y[mask], minlength=256)[low : high + 1]
        cdf = hist.cumsum()
        nonzero = np.flatnonzero(cdf)
        if len(nonzero) == 0 or cdf[-1] == cdf[nonzero[0]]:
            output[mask] = y[mask]
            continue
        cdf_min = cdf[nonzero[0]]
        lut = np.round((cdf - cdf_min) * (high - low) / (cdf[-1] - cdf_min) + low)
        lut = np.clip(lut, low, high).astype(np.uint8)
        output[mask] = lut[y[mask] - low]
    ycrcb[..., 0] = output
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
```

Keep the BBHE implementation luminance-only and preserve the two original chroma channels.

- [x] **Step 6: Run focused tests and the baseline regression suite**

Run: `python3 -m pytest tests/test_member1_processing.py -q`

Expected: all focused tests pass.

Run: `python3 -m pytest -q`

Expected: all existing baseline tests plus the new tests pass.

- [x] **Step 7: Commit the pure processing package**

```bash
git add requirements.txt pyproject.toml src/hripcb_member1 tests/test_member1_processing.py
git commit -m "feat: add Member 1 Gaussian and BBHE processing"
```

---

### Task 2: Add image-quality metrics and deterministic output naming

**Files:**
- Create: `src/hripcb_member1/metrics.py`
- Create: `tests/test_member1_metrics.py`

**Interfaces:**
- `derive_variant_seed(global_seed: int, relative_name: str, variant: str) -> int`
- `calculate_psnr(reference: np.ndarray, candidate: np.ndarray) -> float`
- `calculate_ssim(reference: np.ndarray, candidate: np.ndarray) -> float`
- `variant_name(prefix: str, value: float) -> str`

- [x] **Step 1: Write failing metric tests**

```python
def test_identical_images_have_infinite_psnr_and_unit_ssim():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert calculate_psnr(image, image) == float("inf")
    assert calculate_ssim(image, image) == pytest.approx(1.0)


def test_variant_seed_and_names_are_stable():
    assert derive_variant_seed(42, "01_missing_hole_06.jpg", "sigma30") == derive_variant_seed(42, "01_missing_hole_06.jpg", "sigma30")
    assert variant_name("sigma", 30) == "sigma30"
    assert variant_name("alpha", 0.5) == "alpha050"
```

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_member1_metrics.py -q`

Expected: FAIL because the metrics module does not exist.

- [x] **Step 3: Implement metrics and stable naming**

Use SHA-256 bytes for stable seed derivation rather than Python's process-randomized `hash()`. Calculate PSNR from the BGR mean squared error. Calculate SSIM with `skimage.metrics.structural_similarity(channel_axis=2, data_range=255)`, returning a float.

- [x] **Step 4: Run metric tests**

Run: `python3 -m pytest tests/test_member1_metrics.py -q`

Expected: PASS.

- [x] **Step 5: Commit metrics**

```bash
git add src/hripcb_member1/metrics.py tests/test_member1_metrics.py
git commit -m "feat: add Member 1 quality metrics"
```

---

### Task 3: Build the batch comparison runner

**Files:**
- Create: `configs/member1.yaml`
- Create: `src/hripcb_member1/runner.py`
- Create: `scripts/run_member1_comparison.py`
- Create: `tests/test_member1_runner.py`
- Create: `tests/test_member1_cli.py`

**Interfaces:**
- `load_member1_config(path: Path) -> dict`
- `run_comparison(dataset_root: Path, output_root: Path, sample_name: str | None, config: dict) -> Path`
- CLI: `python3 scripts/run_member1_comparison.py --dataset HRIPCB_UPDATE --output runs/member1 --config configs/member1.yaml`

- [x] **Step 1: Write failing runner tests**

```python
def test_runner_processes_fixture_images_without_touching_sources(tmp_path):
    dataset = make_test_dataset(tmp_path / "dataset", count=2)
    before = {path: path.read_bytes() for path in dataset.glob("*.jpg")}
    output = run_comparison(dataset.parent, tmp_path / "output", None, config)
    assert (output / "image_metrics.csv").is_file()
    assert (output / "processing_times.csv").is_file()
    assert {path: path.read_bytes() for path in dataset.glob("*.jpg")} == before
```

- [x] **Step 2: Run the runner test and verify failure**

Run: `python3 -m pytest tests/test_member1_runner.py -q`

Expected: FAIL because the runner does not exist.

- [x] **Step 3: Add the fixed configuration**

`configs/member1.yaml` must contain:

```yaml
dataset: HRIPCB_UPDATE
split: test
seed: 42
noise_sigmas: [15, 30, 50]
contrast_alphas: [0.75, 0.5, 0.25]
visual_noise_sigma: 30
visual_contrast_alpha: 0.5
gaussian_kernel_size: 5
gaussian_sigma_x: 1.0
jpeg_quality: 95
```

- [x] **Step 4: Implement deterministic batch processing**

Process files in sorted order. For each source image:

1. Read and validate BGR `uint8` data.
2. Save a clean reference copy only under `runs/member1/images/original/`.
3. For each noise sigma, create `noisy_sigmaNN` and `gaussian_sigmaNN` variants.
4. For each contrast alpha, create `low_contrast_alphaAAA` and `bbhe_alphaAAA` variants.
5. Create the medium combined variant from noise sigma 30 and contrast alpha 0.5.
6. Save each result at the original dimensions and JPEG quality 95.
7. Record source filename, variant, parameters, PSNR, SSIM, and elapsed milliseconds.

The source file must be read-only from the runner's perspective; never write into `HRIPCB_UPDATE`.

- [x] **Step 5: Implement CSV and manifest outputs**

Write:

- `image_metrics.csv`: one row per source/variant with `source`, `variant`, `noise_sigma`, `contrast_alpha`, `psnr`, and `ssim`.
- `processing_times.csv`: one row per source/variant with `source`, `variant`, and `milliseconds`.
- `run_manifest.json`: config, source count, selected sample, output directories, and package versions.

- [x] **Step 6: Run the fixture test and full test suite**

Run: `python3 -m pytest tests/test_member1_runner.py -q`

Expected: PASS.

Run: `python3 -m pytest -q`

Expected: all tests pass.

- [x] **Step 7: Commit the runner**

```bash
git add configs/member1.yaml src/hripcb_member1/runner.py scripts/run_member1_comparison.py tests/test_member1_runner.py
git commit -m "feat: add Member 1 batch comparison runner"
```

---

### Task 4: Generate the representative comparison page

**Files:**
- Create: `src/hripcb_member1/report.py`
- Modify: `src/hripcb_member1/runner.py`
- Create: `tests/test_member1_report.py`

**Interfaces:**
- `build_comparison_grid(images: dict[str, np.ndarray], output_path: Path) -> Path`
- `write_comparison_html(output_dir: Path, context: dict) -> Path`

- [x] **Step 1: Write failing report tests**

```python
def test_report_contains_all_required_panel_labels(tmp_path):
    html_path = write_comparison_html(tmp_path, {
        "source": "sample.jpg",
        "panels": [
            {"label": "Original", "src": "original.jpg"},
            {"label": "Noisy", "src": "noisy.jpg"},
            {"label": "Gaussian Filtering", "src": "gaussian.jpg"},
            {"label": "Low Contrast", "src": "low.jpg"},
            {"label": "BBHE", "src": "bbhe.jpg"},
            {"label": "Gaussian + BBHE", "src": "combined.jpg"},
        ],
    })
    html = html_path.read_text()
    assert "Gaussian Filtering" in html
    assert "BBHE" in html
    assert "Gaussian + BBHE" in html
```

- [x] **Step 2: Run the report test and verify failure**

Run: `python3 -m pytest tests/test_member1_report.py -q`

Expected: FAIL because the report module does not exist.

- [x] **Step 3: Implement a six-panel grid and HTML page**

The page must show the fixed representative sample in this order:

```text
Original | Noisy (sigma=30) | Gaussian Filtering
Low Contrast (alpha=0.50) | BBHE | Gaussian + BBHE
```

Include parameter badges, source filename, a short explanation of each transformation, and links to the full-resolution output files. Use relative paths so the page works from the output directory without a web framework.

- [x] **Step 4: Connect the report to the runner**

Select the first sorted test image by default; allow `--sample <filename>` to override it. Save `comparison/comparison_grid.jpg`, `comparison/comparison.html`, and `comparison/representative_manifest.json`.

- [x] **Step 5: Run report tests and full tests**

Run: `python3 -m pytest tests/test_member1_report.py -q`

Expected: PASS.

Run: `python3 -m pytest -q`

Expected: all tests pass.

- [x] **Step 6: Commit the report**

```bash
git add src/hripcb_member1/report.py src/hripcb_member1/runner.py tests/test_member1_report.py
git commit -m "feat: add Member 1 visual comparison report"
```

---

### Task 5: Run the real Member 1 experiment and verify handoff

**Files:**
- Generate: `runs/member1/`
- Modify: `README.md`

- [x] **Step 1: Run the real batch comparison**

Run:

```bash
python3 scripts/run_member1_comparison.py \
  --dataset HRIPCB_UPDATE \
  --output runs/member1 \
  --config configs/member1.yaml
```

Expected: all 70 test images are processed; no source image changes; the HTML page and CSV/JSON summaries exist.

- [x] **Step 2: Verify output counts and source immutability**

Run:

```bash
python3 - <<'PY'
import csv, json
from pathlib import Path

root = Path("runs/member1")
manifest = json.loads((root / "run_manifest.json").read_text())
assert manifest["source_count"] == 70
assert (root / "comparison/comparison.html").is_file()
with (root / "image_metrics.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 70 * 15
print("member1 output verification: OK")
PY
```

- [x] **Step 3: Update README with Member 1 commands and outputs**

Document the exact command, representative page path, degradation parameters, and the fact that `best.pt` is intentionally not used in this phase.

- [x] **Step 4: Run the complete regression suite**

Run: `python3 -m pytest -q`

Expected: all baseline and Member 1 tests pass.

- [x] **Step 5: Commit implementation and documentation**

```bash
git add README.md configs/member1.yaml src/hripcb_member1 scripts/run_member1_comparison.py tests
git commit -m "feat: complete Member 1 Gaussian BBHE comparison"
```
