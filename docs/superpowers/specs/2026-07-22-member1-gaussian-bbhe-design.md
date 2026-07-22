# Member 1 Gaussian Filtering and BBHE Design

## Goal

Create a reproducible, image-only comparison module for Member 1 that demonstrates Gaussian noise removal and BBHE contrast enhancement on HRIPCB test images before any YOLO inference is added.

## Scope

This phase does not load, modify, retrain, or evaluate `runs/baseline/weights/best.pt`. It only creates degraded images, filtered/enhanced images, visual comparisons, and image-quality/timing summaries.

The module will use the supplied 70-image HRIPCB test split. One deterministic representative image will be shown in a browser-viewable HTML page, while all 70 images will be processed and recorded for later detector evaluation.

## Input degradation contract

HRIPCB images are relatively clean and have no labelled noise or contrast-degradation levels. The module will therefore create controlled inputs from each clean image.

- Work in OpenCV BGR images on disk.
- Convert to YCrCb for degradation and processing so that only luminance changes; preserve chroma channels.
- Gaussian noise is added to the Y channel using a seeded normal distribution, clipped to `[0, 255]` and converted back to `uint8`.
- Noise levels are `sigma=15`, `30`, and `50`.
- Low contrast is created by `Y_low = clip(128 + alpha * (Y - 128), 0, 255)`.
- Contrast levels are `alpha=0.75`, `0.50`, and `0.25`.
- The representative visual comparison uses `sigma=30` and `alpha=0.50`.
- The combined degraded input applies Gaussian noise first and contrast reduction second. The combined processing pipeline applies Gaussian Filtering first and BBHE second.
- A deterministic per-image seed is derived from the global seed, relative image path, and degradation level so reruns and future member modules use identical degraded inputs.

## Processing contract

### Gaussian Filtering

Apply OpenCV Gaussian blur to the complete BGR image using a fixed `5x5` kernel and `sigmaX=1.0`. This is intentionally a simple, low-cost baseline and is not tuned per image.

### BBHE

Implement Brightness Preserving Bi-Histogram Equalization on the Y channel:

1. Calculate the mean luminance.
2. Split the histogram into `[0, mean]` and `[mean+1, 255]` ranges.
3. Equalize each range independently with a CDF mapping constrained to its original range.
4. Merge the transformed Y channel with the original Cr/Cb channels.

Constant or empty histogram ranges must be handled without division-by-zero or invalid pixels.

## Output contract

Outputs are written under `runs/member1/` and do not replace baseline outputs:

```text
runs/member1/
├── comparison/
│   ├── comparison.html
│   ├── comparison_grid.jpg
│   └── representative_manifest.json
├── images/
│   ├── noisy_sigma15/
│   ├── gaussian_sigma15/
│   ├── low_contrast_alpha075/
│   ├── bbhe_alpha075/
│   └── ...
├── image_metrics.csv
├── processing_times.csv
└── run_manifest.json
```

The representative page shows Original, Noisy, Gaussian output, Low-contrast, BBHE output, and Gaussian+BBHE output. It includes the exact parameters and source filename. All 70 test images are saved for later detection evaluation, and each output is linked to its source image through CSV and JSON manifests.

Image-quality summaries compare each generated result with the clean source using PSNR and SSIM. These are secondary visual-quality measures; detector metrics will be added in a later phase using the frozen shared checkpoint.

## Fairness and safety constraints

- Use only the existing test image files; do not change images or labels.
- Keep all four members' degradation levels, random seed, source split, and output naming compatible.
- Do not use `best.pt` in this phase.
- Never overwrite `runs/baseline/` or `runs/evaluation/`.
- Preserve original image dimensions and use JPEG quality 95 for generated images.
- Record parameters, package versions, source count, and output paths in `run_manifest.json`.

## Acceptance criteria

1. Unit tests cover deterministic degradation, pixel bounds, Gaussian filtering, BBHE output shape/dtype, and constant-image safety.
2. The CLI processes all 70 test images without changing the source files.
3. The CLI writes all generated variants, a representative HTML comparison, metrics CSV, timing CSV, and a manifest.
4. The HTML page can be opened locally and shows all six agreed visual panels.
5. Existing baseline tests continue to pass and the shared checkpoint remains untouched.
