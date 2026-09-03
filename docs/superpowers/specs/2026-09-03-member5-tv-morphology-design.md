# Member 5 TV + Top-hat/Black-hat Design

## Goal

Add a fifth preprocessing module that combines Total Variation (TV) denoising
with morphological Top-hat and Black-hat enhancement. The search must measure
detector performance on the validation split instead of inferring detection
quality from SSIM or PSNR.

The implementation delivers a resumable script that the user will run manually.
Implementation and automated tests do not start the full parameter sweep.

## Technique Definition

Member 5 contains two conceptual stages:

1. TV denoising removes noise while preserving edges.
2. One morphological enhancement stage uses both Top-hat and Black-hat to
   emphasize small bright and dark structures respectively.

The combined pipeline order is fixed:

```text
denoised = TV(original, weight)
top = TopHat(denoised, kernel)
black = BlackHat(denoised, kernel)
output = denoised + top_amount * top - black_amount * black
```

Top-hat and Black-hat are computed independently from the same denoised image;
neither operation consumes the output of the other. The morphology-only control
uses the same equation with the original image as its base.

Morphology is applied to luminance while preserving chroma. The structuring
element is elliptical and shared by Top-hat and Black-hat. Arithmetic is
performed in a signed or floating-point representation and clipped to `uint8`
only after both contributions are combined.

## Search Matrix

The approved search is deliberately smaller than the original 368-candidate
proposal while retaining weak, medium, and strong settings.

```yaml
tv_weights: [0.01, 0.02, 0.05]
morphology_kernel_sizes: [5, 9, 15]
top_hat_amounts: [0.5, 1.0]
black_hat_amounts: [0.5, 1.0]
```

This produces:

- 1 original control;
- 3 TV-only controls;
- 12 Top-hat/Black-hat-only controls (`3 x 2 x 2`);
- 36 mandatory combined candidates (`3 x 3 x 2 x 2`);
- 52 candidates in total.

The final Member 5 winner is selected only from the 36 combined candidates.
Every eligible winner therefore executes TV, Top-hat, and Black-hat with
non-zero strength.

## TV Parameters

Use `skimage.restoration.denoise_tv_chambolle` with RGB channel semantics and
normalized floating-point input. Only `weight` is an experiment parameter.
`eps`, `max_num_iter`, and `channel_axis` remain fixed because they control
convergence or data interpretation rather than defining the accuracy search.
The implementation does not mix Chambolle and Bregman TV variants in one grid.

## Candidate and Result Schema

The public technique identifiers are:

- `tv` for TV-only controls;
- `top_black_hat` for morphology-only controls;
- `tv_top_black_hat` for mandatory combined candidates.

Member 5 records store these reproducibility parameters where applicable:

- `tv_weight`;
- `morphology_kernel_size`;
- `top_hat_amount`;
- `black_hat_amount`.

Candidate IDs encode every varying parameter and must be unique. Records retain
the existing checkpoint, model, split, evaluation type, image-quality metrics,
detection metrics, source count, and preview fields used by the shared runner.

## Resumable Search Script

Add `scripts/run_member5_full_search.py` and
`configs/member5_full_search.yaml`. The normal invocation is:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_member5_full_search.py
```

The script processes a bounded batch of candidates at a time. Its default batch
size is small enough to avoid retaining the entire generated dataset, and it
accepts `--batch-size` for local adjustment.

After each batch:

1. Finish frozen-checkpoint YOLO evaluation.
2. Atomically persist the completed records and progress state.
3. Retain parameters, metrics, ranked data, and one preview per candidate.
4. Delete the batch's full processed images and evaluation staging files.

Deletion occurs only after durable result persistence. If execution stops during
a batch, the next run discards or replaces that incomplete batch and recomputes
it. Completed candidate IDs are skipped automatically.

The progress state includes a deterministic fingerprint of all result-affecting
configuration. A changed configuration cannot resume into an existing output
directory. The script exits with a clear message instructing the user to choose
a new output directory. `--keep-variants` disables post-evaluation cleanup when
full processed datasets are intentionally required.

The completed output is:

```text
runs/member5_full_search/
├── results.json
├── results.csv
├── summary.json
├── dashboard.html
├── progress.json
└── previews/
```

`summary.json` records the best TV-only control, the best morphology-only
control, the best mandatory combination, its difference from the shared
original baseline, and ranked combined candidates.

The implementation must not launch the full sweep. The user will run the script
after reviewing the command and expected output.

## Evaluation and Selection

- Use only the existing `val` split for tuning and selection.
- Keep the shared baseline YOLO checkpoint and its image size, confidence, IoU,
  device selection, and worker configuration unchanged.
- Rank combined candidates by `mAP50-95`.
- Report `mAP50`, precision, recall, F1, mean SSIM, mean PSNR, and processing
  time as secondary diagnostics.
- Never select a winner from SSIM or PSNR alone.
- Do not use the frozen test split during this sweep.
- A later test evaluation or retraining run is a separate task after the winner
  has been frozen.

## Project Dashboard and Streamlit Integration

Extend the shared application from four to five member modules:

- add Member 5 labels for all three technique identifiers;
- recognize `tv_top_black_hat` as a combined technique;
- include `member5` in shared-original collapsing and best-by-module analysis;
- include Member 5 in project aggregation and selection;
- update text that currently says "four members" or "Member 1-4";
- expose Member 5 records through comparison filters, analysis tables, image
  inference, and video inference;
- display all four Member 5 parameters in the existing parameter panel.

The completed Member 5 script merges its 52 final records into
`runs/project_validation_comparison/results.json` without removing existing
Member 1-4 or official comparison records. Partial Member 5 results never enter
the project-wide recommendation or leaderboard.

Generated sweep data remains ignored by Git. The tracked project results JSON
is updated only when the complete sweep succeeds. The script does not commit or
push. Publishing the updated hosted Streamlit application remains a separate,
explicit Git action by the user.

## Code Boundaries

Expected production changes are limited to:

- `src/hripcb_preprocessing/filters.py` for TV and morphology primitives;
- `src/hripcb_preprocessing/candidates.py` for Member 5 candidate construction
  and dispatch;
- the generic runner or a focused batching helper for explicit candidate batches,
  atomic progress, resume validation, and safe cleanup;
- `configs/member5_full_search.yaml` for the approved 52-candidate matrix;
- `scripts/run_member5_full_search.py` for orchestration and summary output;
- dashboard analysis, filtering, static dashboard, aggregation, and Streamlit
  files required to recognize a fifth member;
- focused tests and concise README usage documentation.

Unrelated preprocessing math, existing member grids, model training defaults,
and official test results remain unchanged.

## Error Handling and Cleanup Safety

Reject invalid images, non-positive TV weights, non-positive or even kernel
sizes, negative amounts, unsupported technique names, and malformed configs
before expensive processing begins.

Cleanup targets must be resolved below the active Member 5 batch directory.
Never delete the output root, results, progress state, summaries, or previews.
An evaluation or result-write failure leaves enough state to rerun the incomplete
batch and never marks it complete.

Missing dataset images, labels, checkpoint, or data configuration fail with an
explicit path and do not modify the project-wide results JSON.

## Testing

Implementation follows red-green-refactor. Required tests cover:

1. TV preserves image shape and `uint8` output and rejects invalid weights.
2. Top-hat/Black-hat preserves shape and dtype and rejects invalid kernels and
   amounts.
3. Top-hat and Black-hat are computed from the same input image.
4. The combined dispatch exactly matches manual `TV -> morphology` staging.
5. Repository configuration produces 52 total candidates and 36 combined
   candidates with unique IDs and non-zero use of all required operations.
6. Batch progress is durable and completed candidates are skipped on resume.
7. Changed result-affecting configuration is rejected during resume.
8. Intermediate variant data is deleted only after results are persisted, while
   `--keep-variants` preserves it.
9. Partial runs do not update the project-wide results JSON.
10. Final aggregation preserves Member 1-4 records and adds Member 5.
11. Dashboard filtering, combined selection, analysis, image inference, and
    video inference recognize Member 5.
12. Existing preprocessing and dashboard tests continue to pass.

Before handoff, run the focused tests, the complete test suite, compile checks,
and `git diff --check`. Do not use the full 52-candidate sweep as an automated
verification step.

## Out of Scope

- Running the Member 5 full sweep on the user's behalf.
- Choosing or claiming a winning score before the sweep finishes.
- Tuning on the test split.
- Retraining the detector.
- Replacing or retuning Members 1-4.
- Automatically committing, pushing, or deploying generated results.
