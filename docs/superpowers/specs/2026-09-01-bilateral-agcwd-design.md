# Member 3 Canonical AGCWD Combination Design

## Goal

Improve the mandatory two-method Member 3 pipeline while preserving an honest
comparison against the shared YOLO baseline. Every selectable combined candidate
must apply both Bilateral Filtering and AGCWD with non-zero strength.

The validation acceptance gate is `mAP50-95 > 0.5151173658` (the saved original
input baseline). Matching or exceeding the current Bilateral-only result
`0.5235315781` is the stretch goal. `mAP50`, precision, recall, F1, image-quality
metrics, per-class results when available, and processing time remain secondary
diagnostics.

## Constraints

- The assignment requires both Bilateral Filtering and AGCWD.
- A combined candidate must use `agcwd_blend >= 0.05`; zero-strength candidates
  are controls only and cannot win the combined leaderboard.
- Parameters are selected exclusively on `val`. The frozen `test` split is not
  used during development or tuning.
- The frozen baseline checkpoint, image size, confidence threshold, IoU threshold,
  worker count, and dataset split remain unchanged during the ablation.
- Existing Member 1, Member 2, and Member 4 behavior is out of scope.
- Full detector validation requires the untracked HRIPCB image splits to be
  present locally. Unit and integration tests must not depend on those images.

## Selected Approach

Replace the current project-specific `AGCWD-style` luminance transform with a
paper-aligned AGCWD implementation, then control its contribution with an
explicit blend. This is preferred over merely blending the current transform
because it gives the report a defensible relationship to Huang, Cheng, and
Chiu's 2013 AGCWD method.

The implementation will:

1. Convert BGR input to HSV.
2. Compute the normalized histogram of the `V` channel.
3. Apply the paper's weighted distribution using an explicit `alpha` parameter.
4. Build the weighted CDF and adaptive gamma map `1 - CDF`.
5. Transform only `V`, preserving `H` and `S`.
6. Convert the result back to BGR.

The existing global `gamma` multiplier is retired from new Member 3 candidates.
Generated records use `agcwd_alpha`, `agcwd_blend`, and `agcwd_order`, so results
cannot be confused with the previous search.

## Pipeline Semantics

Both supported orders use blending at the point where AGCWD is introduced.

### Bilateral then AGCWD

```text
bilateral = Bilateral(original)
enhanced = AGCWD(bilateral, alpha)
output = blend(bilateral, enhanced, agcwd_blend)
```

### AGCWD then Bilateral

```text
enhanced = AGCWD(original, alpha)
soft_enhanced = blend(original, enhanced, agcwd_blend)
output = Bilateral(soft_enhanced)
```

For every combined candidate, `agcwd_blend` is strictly positive, so AGCWD
changes the data passed to the output. Bilateral is also always executed.
Blending uses a deterministic weighted sum with clipping and `uint8` output.

## Candidate Search

The first search is intentionally conservative:

```yaml
bilateral_presets:
  - b05: diameter 5, sigma_color 25, sigma_space 25
  - b07: diameter 7, sigma_color 50, sigma_space 50
  - b09: diameter 9, sigma_color 75, sigma_space 75
agcwd_alphas: [0.25, 0.5, 0.75, 1.0]
agcwd_blends: [0.05, 0.1, 0.15, 0.2, 0.3]
agcwd_orders: [bilateral_then_agcwd, agcwd_then_bilateral]
```

The matrix contains 120 combined candidates (`3 x 4 x 5 x 2`), plus the original,
Bilateral-only, and canonical AGCWD-only controls. All candidates are evaluated
under the existing shared runner so source images, checkpoint, thresholds, and
metrics stay comparable.

If runtime is prohibitive, the same implementation may be run in two documented
stages: first fix `b07` and scan all AGCWD settings and orders; then scan all three
Bilateral presets using the two strongest AGCWD settings. The final comparison
must still evaluate the shortlisted candidates on the complete validation split.

## Code Changes

### `src/hripcb_preprocessing/filters.py`

- Change `apply_agcwd` to accept the paper's `alpha` parameter.
- Apply AGCWD to HSV `V`, not YCrCb luminance.
- Validate `0 < alpha <= 1`.
- Add a small, independently tested image blend helper that validates
  `0 <= strength <= 1`, identical shapes, and `uint8` input.

### `src/hripcb_preprocessing/candidates.py`

- Build Member 3 AGCWD controls from `agcwd_alphas`.
- Build combined candidates from the Cartesian product of Bilateral presets,
  alphas, positive blends, and the two supported orders.
- Keep the public technique name `bilateral_agcwd` for dashboard compatibility;
  store order in `parameters["agcwd_order"]` and candidate IDs.
- Dispatch both order semantics exactly as defined above.
- Reject unknown orders and reject combined candidates with zero blend.

### `configs/member3_sweep.yaml`

- Replace `agcwd_gammas` with `agcwd_alphas`, `agcwd_blends`, and
  `agcwd_orders`.
- Keep `split: val` and all shared evaluation settings unchanged.

### Documentation

- Update the Member 3 sweep description and result wording so it distinguishes
  the old aggressive combination from the new canonical/blended search.
- Do not publish a new score until a full validation run has generated it.

## Compatibility and Result Integrity

Old JSON result files remain readable because their parameter dictionaries are
treated as stored experiment records rather than replayed configuration. New
candidate generation does not silently reinterpret `agcwd_gamma`; a stale Member
3 config fails with a clear missing-key error instead of producing mislabeled
results.

The dashboard continues to recognize `bilateral_agcwd` as a combined technique.
Candidate IDs encode order, alpha, and blend, preventing collisions and making
the winning pipeline reproducible.

## Tests

Implementation follows red-green-refactor. Tests are written and observed failing
before production changes.

Required tests:

1. Canonical AGCWD preserves shape and `uint8` dtype.
2. `alpha` outside `(0, 1]` is rejected.
3. AGCWD preserves HSV hue and saturation within conversion-rounding tolerance.
4. Blend strength `0` returns the base image and `1` returns the enhanced image.
5. Invalid blend inputs and strengths are rejected.
6. Member 3 candidate counts match the configured Cartesian product.
7. Every combined candidate contains positive blend, explicit alpha, and order.
8. `bilateral_then_agcwd` matches a manually staged reference pipeline.
9. `agcwd_then_bilateral` matches a manually staged reference pipeline.
10. Unknown order and zero-blend combined candidates are rejected.
11. Existing preprocessing, dashboard, runner, and configuration tests continue
    to pass.

## Validation and Selection

After automated tests pass and the HRIPCB validation images are available:

1. Run the Member 3 validation sweep with the frozen checkpoint.
2. Rank combined candidates by `mAP50-95` only.
3. Compare the winner with original baseline and Bilateral-only control.
4. Inspect `mAP50`, precision, recall, F1, PSNR, SSIM, runtime, and available
   per-class metrics for regressions hidden by the aggregate.
5. Visually inspect representative images from all six defect classes for lost
   copper edges, clipped highlights, color shifts, or obscured small defects.
6. Accept the combined candidate only if its validation `mAP50-95` exceeds the
   baseline and no critical category has an unacceptable regression.
7. If no positive-blend candidate passes, report that the mandatory combination
   is the best under the assignment constraint but is not better than the model's
   original input. Do not tune on `test` to force a positive result.
8. Retraining with the selected deterministic pipeline and the official frozen
   test evaluation are separate follow-up stages, not part of this change.

## Error Handling

Invalid images, alpha values, blend values, or order names raise `ValueError`
before any expensive sweep work begins. Missing dataset images or labels continue
to use the runner's existing explicit file errors. No candidate is silently
clamped into the accepted parameter range.

## Out of Scope

- Changing the shared YOLO checkpoint or training configuration.
- Tuning on the test split.
- Replacing Bilateral Filtering or AGCWD with a different technique.
- Claiming that the new pipeline reaches a target score before the full validation
  run exists.
- Changes to the other three member modules.
