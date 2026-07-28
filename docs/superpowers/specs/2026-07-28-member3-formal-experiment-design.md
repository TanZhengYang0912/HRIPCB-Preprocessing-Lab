# Member 3 Formal Experiment Design

## Goal

Produce a reproducible Member 3 validation study that compares the agreed 16
clean-image preprocessing variants with one frozen YOLOv8s baseline. The
study selects a Member 3 candidate by validation mAP50-95 without reading or
ranking test-set metrics.

## Scope

Included:

- one original-image reference condition;
- three Bilateral Filtering presets;
- three AGCWD plus global-gamma presets;
- nine Bilateral Filtering plus AGCWD plus global-gamma presets;
- validation-only detection evaluation, image-quality metrics, preprocessing
  timing, machine-readable result records, and a Member 3 Streamlit view.

Excluded:

- Gaussian-noise experiments from the formal ranking;
- Member 1, Member 2, or Member 4 techniques;
- test-set evaluation during Member 3 parameter selection;
- training `runs/final_model/weights/best.pt`.

The existing noise-based `runs/member3/` outputs remain preliminary artefacts.
Formal outputs are written under `runs/member3_formal/`.

## Fixed Evaluation Contract

| Field | Value |
| --- | --- |
| Dataset | `HRIPCB_UPDATE` |
| Dataset split | `val` |
| Validation images | 138 |
| Checkpoint | `runs/baseline/weights/best.pt` |
| Image size | 1024 |
| Confidence | 0.25 |
| IoU | 0.70 |
| Device | `auto` (`mps` when available, otherwise `cpu`) |
| Workers | 0 |
| Primary ranking metric | `mAP50-95` |

Every formal row includes the fixed settings, model/checkpoint identity,
Member 3 technique, full preprocessing parameters, Precision, Recall, F1,
mAP50, mAP50-95, PSNR, SSIM, and mean preprocessing time in milliseconds.
The runner verifies that the selected validation split contains exactly 138
images before it starts processing.

## Algorithms and Presets

The original AGCWD method uses `alpha` as the weighting-distribution
adjustment parameter. It is not valid to pass the teammate's `1.2` gamma value
to the existing `alpha` argument, which only accepts `(0, 1]`.

For a clear and repeatable experiment, Member 3 keeps the existing AGCWD
weighting parameter fixed at `alpha=0.75`, then applies a standard global
gamma correction to the AGCWD output. The gamma preset is recorded separately
as `gamma` and uses the agreed values `0.8`, `1.0`, and `1.2`.

The 16 formal conditions are:

| Technique | Count | Parameters |
| --- | ---: | --- |
| Original | 1 | no preprocessing |
| Bilateral Filtering | 3 | `(d=5, sigmaColor=25, sigmaSpace=25)`, `(d=7, 50, 50)`, `(d=9, 75, 75)` |
| AGCWD + gamma | 3 | `alpha=0.75`, `gamma in {0.8, 1.0, 1.2}` |
| Bilateral + AGCWD + gamma | 9 | every Bilateral preset crossed with every gamma preset |

All preprocessing operates on Y in YCrCb; Cr and Cb are preserved. Global
gamma correction uses `output = 255 * (input / 255) ** gamma` with rounding
and clipping to `uint8`.

The implementation adds `scikit-image` for its standard structural-similarity
calculation. PSNR and SSIM compare each processed image to its corresponding
original validation image; they describe image fidelity rather than detection
accuracy.

## Result Record

Each CSV/JSON row uses these fields:

```text
model_id, checkpoint, member, technique, training_preprocessing,
evaluation_preprocessing, dataset_split, validation_images, imgsz, conf, iou,
device, workers, bilateral_diameter, bilateral_sigma_color,
bilateral_sigma_space, agcwd_alpha, gamma, precision, recall, f1, map50,
map50_95, psnr, ssim, processing_time_ms
```

For Original, bilateral and AGCWD parameter fields are empty, `gamma=1.0`,
`psnr=inf`, `ssim=1.0`, and `processing_time_ms=0.0`. The runner uses
`map50_95` descending as the only formal ranking key and breaks exact ties by
the stable condition identifier.

## Data Flow

1. Validate the requested dataset root and the frozen checkpoint.
2. For every condition, create a YOLO-compatible validation-only processed
   dataset under `runs/member3_formal/processed/` without modifying the source
   dataset.
3. Measure preprocessing time for every image and aggregate its mean.
4. Compute PSNR and SSIM between the original validation image and its
   processed counterpart. Original has `PSNR=inf`, `SSIM=1.0`, and
   `processing_time_ms=0.0`.
5. Evaluate the frozen checkpoint with the fixed validation settings.
6. Write one normalized result record per condition to JSON and CSV, sorted by
   descending mAP50-95.
7. Write `summary.json` with the winning Member 3 candidate and exact run
   contract.

## UI Behaviour

The Member 3 dashboard reads the formal result CSV when present. It replaces
the noise-sigma control with a technique control, an applicable Bilateral
preset control, and an applicable gamma control. Its default summary view
shows the exact matching formal condition. A separate control reveals all 16
validation experiments for comparison. The UI labels the results as
`Validation tuning`, shows the current selection and fixed evaluation settings,
and never combines these records with the old preliminary test CSV.

## Error Handling and Verification

- The formal runner stops before evaluation if any required validation
  image/label directory or the checkpoint is absent.
- Dataset preparation fails on unreadable images or missing matching labels.
- Unit tests cover the formal 16-condition matrix, gamma correction,
  validation-only filtering, PSNR/SSIM aggregation, ranking, and CSV schema.
- A smoke test verifies CLI argument parsing and script compilation.
- The actual full experiment is run only when the complete `HRIPCB_UPDATE`
  dataset is available locally.
