# Member 3 Streamlit Demo Design

## Goal

Add a local Streamlit dashboard that lets a user upload one PCB image, apply
the frozen Member 3 test-time preprocessing pipeline, and view YOLOv8s defect
detections. The dashboard is a visual demonstration of the existing
experiment; it must not retrain or modify the shared detector.

## Confirmed user experience

- Run locally with `streamlit run scripts/member3_demo.py`.
- Upload one `.jpg`, `.jpeg`, or `.png` image.
- Select one condition: `Clean`, `Noisy`, `Bilateral Filtering`, `AGCWD`, or
  `Bilateral + AGCWD`.
- Select Gaussian-noise sigma `10`, `25`, or `40` when relevant.
- Use the tuned Member 3 parameters: bilateral `d=7`, `sigmaColor=75`,
  `sigmaSpace=75`, and AGCWD `alpha=0.5`.
- Keep YOLO inference at `imgsz=1024`, with a default confidence threshold of
  `0.25` and an adjustable visualisation-only threshold.
- Show Original Image, Processed Image, and Detection Result side by side.
- Show detected class names, confidence, box count, and inference time.
- If the uploaded file can be matched to a dataset image with a label file,
  show Ground Truth boxes as a fourth visual reference; otherwise omit it.
- Provide a Compare All Conditions action for the same image, while keeping
  single-condition inference as the default path.
- Save original, processed, prediction, and JSON metadata under
  `runs/member3_demo/`.
- Show the existing `runs/member3/comparison.csv` as an experiment-summary
  table, without recomputing dataset-level mAP for a single uploaded image.
- Keep defect class names exactly as defined by the dataset:
  `Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spurious_copper`,
  and `Spur`.

## Architecture

1. `src/hripcb_baseline/member3_demo.py` provides pure, UI-independent seams:
   - map UI condition names to the existing preprocessing functions;
   - prepare an RGB image using the existing Member 3 parameters;
   - run the frozen Ultralytics model and convert detections into serialisable
     records;
   - render prediction and optional ground-truth boxes;
   - create a timestamped output directory and JSON metadata.
2. `scripts/member3_demo.py` contains the Streamlit UI only. It loads the
   checkpoint from `runs/baseline/weights/best.pt`, calls the demo seams, and
   renders images, controls, detection records, and the CSV summary.
3. `requirements.txt` adds the Streamlit runtime dependency.

The demo reuses the existing implementation in `member3.py` and the same
fixed detector settings as the batch experiment. It does not duplicate the
algorithm or create a second model checkpoint.

## Data flow

```text
uploaded image
    -> RGB image
    -> selected Y-channel preprocessing
    -> YOLOv8s best.pt at imgsz=1024
    -> detections and annotated image
    -> UI + runs/member3_demo/YYYYMMDD-HHMMSS/
```

For `Bilateral + AGCWD`, processing order is Bilateral Filtering followed by
AGCWD on the Y channel in YCrCb, preserving Cr/Cb. Gaussian noise is applied
reproducibly to Y only, using the same sigma-to-seed mapping as the experiment.

## Error handling

- Missing checkpoint: show an actionable Streamlit error with the expected
  path.
- Unsupported or unreadable upload: show a clear file-format error.
- Missing CSV: render the dashboard without the summary and explain that the
  batch experiment must be run first.
- Missing dataset label match: continue with prediction-only mode.
- Inference exceptions: show a concise error and preserve the uploaded image
  for diagnosis; do not expose a traceback as the main UI.

## Testing seams

Tests will cover public behavior at these boundaries:

- each UI condition produces the expected preprocessing path and RGB output;
- noise is deterministic for a fixed sigma and seed;
- detection records contain class, confidence, and xyxy coordinates;
- annotation preserves image dimensions and draws predictions;
- optional dataset labels are loaded only when a matching label exists;
- saved demo metadata contains condition, sigma, threshold, model path, and
  detections;
- CSV summary loading handles the existing result format.

The Streamlit page itself will receive a smoke import/compile check rather
than browser automation. The existing Member 3 test suite must remain green.

## Acceptance criteria

The feature is ready when a user can run the documented command, upload a
PCB image, choose a Member 3 condition, see preprocessing and YOLO boxes, and
find the saved artefacts under `runs/member3_demo/`; all new tests pass and
the existing test suite has no new failures.
