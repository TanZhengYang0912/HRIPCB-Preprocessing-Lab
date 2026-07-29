# HRIPCB Preprocessing Lab

A shared PCB defect-detection experiment and demonstration prototype. The project uses the HRIPCB dataset and YOLOv8s to compare image-preprocessing techniques across four member modules with one fixed evaluation protocol.

The workflow has two stages:

1. Compare preprocessing techniques with the same frozen baseline checkpoint.
2. Select the best validation result, then retrain a final YOLO model with the same training configuration as the baseline.

## 1. Scope

The dataset contains six PCB defect classes: `Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spurious_copper`, and `Spur`.

| Module | Noise removal | Contrast enhancement |
|---|---|---|
| member1 | Gaussian Filtering | BBHE |
| member2 | Median Filtering | CLAHE |
| member3 | Bilateral Filtering | AGCWD |
| member4 | Non-local Means | Multi-Scale Retinex |

`baseline` is the shared control model, not a fifth member module.

## 2. Repository Structure

```text
ImageProcessing-Assignment/
├── HRIPCB_UPDATE/                 # Local dataset; image splits are not tracked
├── configs/                       # Dataset, baseline and member sweep configs
├── scripts/                       # Training, evaluation, dashboard and report scripts
├── src/                           # Preprocessing, evaluation and dashboard packages
├── tests/                         # Unit, integration and dashboard tests
├── runs/                          # Local models, metrics, reports and videos
├── requirements.txt               # YOLO and computer-vision dependencies
├── requirements-dashboard.txt     # Streamlit dependencies
└── pyproject.toml
```

Dataset images, model weights and generated run outputs are excluded by `.gitignore`. They can be regenerated locally from the source scripts, so large experiment outputs are not accidentally uploaded to GitHub.

## 3. Installation

For macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dashboard.txt
```

Place the dataset in this structure:

```text
HRIPCB_UPDATE/
├── data.yaml
├── train/images/   train/labels/
├── val/images/     val/labels/
└── test/images/    test/labels/
```

Validate the dataset before training or evaluation:

```bash
python3 scripts/validate_dataset.py --root HRIPCB_UPDATE
```

## 4. Frozen Experiment Protocol

Every member comparison must use the same protocol:

| Setting | Value |
|---|---:|
| Dataset | `HRIPCB_UPDATE` |
| Parameter-selection split | `val` |
| Final reporting split | `test` |
| Image size | `1024` |
| Confidence threshold | `0.25` |
| IoU threshold | `0.70` |
| Workers | `0` |
| Training seed | `42` |
| Primary metric | `mAP50-95` |
| Model family | YOLOv8s |

Use `val` to select parameters. Do not tune against `test`; reserve `test` for the final unbiased comparison.

## 5. Shared Baseline Model

All members reuse this checkpoint:

```text
runs/baseline/weights/best.pt
```

The main baseline training configuration was:

```text
model: yolov8s.pt
epochs: 100
imgsz: 1024
batch: 4
patience: 20
seed: 42
workers: 0
device: auto  # Apple MPS is normally selected on the project machine
```

To rebuild the baseline when necessary:

```bash
python3 scripts/train_baseline.py \
  --data configs/hripcb_local.yaml \
  --model yolov8s.pt \
  --project runs \
  --name baseline \
  --epochs 100 \
  --imgsz 1024 \
  --batch 4 \
  --patience 20
```

Do not train a separate detector for every preprocessing technique during the comparison stage. Otherwise, preprocessing and model-training differences become mixed together.

## 6. Baseline Evaluation

```bash
# Validation split
python3 scripts/evaluate_baseline.py \
  --weights runs/baseline/weights/best.pt \
  --data configs/hripcb_local.yaml \
  --split val

# Final test split
python3 scripts/evaluate_baseline.py \
  --weights runs/baseline/weights/best.pt \
  --data configs/hripcb_local.yaml \
  --split test
```

Current saved baseline reference results:

| Split | Precision | Recall | mAP50 | mAP50-95 | F1 |
|---|---:|---:|---:|---:|---:|
| val | 0.9719 | 0.9440 | 0.9420 | 0.5151 | 0.9577 |
| test | 0.9515 | 0.9233 | 0.9208 | 0.4890 | 0.9372 |

Machine-readable metrics are stored in `runs/evaluation/val/metrics.json` and `runs/evaluation/test/metrics.json`.

## 7. Member Validation Sweeps

Run all member sweeps with the shared baseline checkpoint:

```bash
python3 scripts/run_all_validation_sweeps.py \
  --dataset HRIPCB_UPDATE \
  --output-root runs
```

Each sweep changes only preprocessing parameters:

| Module | Candidate parameters |
|---|---|
| member1 | Gaussian kernel `5, 7, 9`; sigmaX `1.0, 1.5, 2.0`; BBHE strength `0.25, 0.5, 0.7, 1.0` |
| member2 | Median kernel `3, 5, 7`; CLAHE clip limit `1.5, 2.0, 3.0`; tile grid `8×8` |
| member3 | Bilateral diameter `5, 7, 9`; sigma colour `25, 50, 75`; AGCWD gamma `0.8, 1.0, 1.2` |
| member4 | NLM `h=3, 7, 10`; MSR scales `(15, 25, 2)`, `(15, 50, 150)`, `(20, 80, 160)` |

The highest current validation result is `member2 / median_k5` with `mAP50-95=0.5269`. This is validation evidence for parameter selection, not the final test score.

## 8. Dashboard, Sorting and Reports

Build the comparison data and static HTML dashboard:

```bash
python3 scripts/build_project_dashboard.py \
  --runs-root runs \
  --output runs/project_validation_comparison

open runs/project_validation_comparison/dashboard.html
```

Start Streamlit:

```bash
streamlit run scripts/streamlit_dashboard.py -- \
  --results runs/project_validation_comparison/results.json
```

The dashboard has four main tabs:

1. **Compare experiments** — filter by model, module, technique and split, then sort by mAP50-95, mAP50, F1, Precision or Recall.
2. **Run image inference** — upload images, select a baseline or final model and technique, and view the original image, preprocessed image and YOLO result.
3. **Analysis & reports** — view experiment count, four member modules, one baseline control, model coverage, ranking and protocol details.
4. **Video processing** — upload a short video and run preprocessing plus frame-by-frame YOLO detection, producing browser-compatible H.264 output when available.

Export a PDF, CSV and JSON report:

```bash
python3 scripts/export_project_report.py \
  --results runs/project_validation_comparison/results.json \
  --output runs/project_report
```

Generated reports are reproducible artifacts. The source code, configs, tests and README are the important files to track in Git; generated images do not need to be committed.

## 9. Final Model Workflow

After selecting the best technique on `val`, preprocess the train, validation and test images, then train a new final YOLO model with the baseline training protocol:

```bash
python3 scripts/prepare_final_dataset.py

python3 scripts/train_final_model.py \
  --weights yolov8s.pt \
  --epochs 100

python3 scripts/evaluate_final_model.py
```

The final model must start from a new YOLO initialization such as `yolov8s.pt`. Do not use the baseline `best.pt` as the starting point for final training.

Current saved official test reference results:

| Model / input | mAP50-95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline YOLO + original | 0.4890 | 0.9515 | 0.9233 | 0.9372 |
| Baseline YOLO + median k=5 | 0.4994 | 0.9528 | 0.9294 | 0.9409 |
| Final YOLO + median k=5 | 0.4879 | 0.9272 | 0.9168 | 0.9220 |

The current evidence shows that `median k=5` performs better than the original input with the frozen baseline checkpoint, while the current final retraining result does not exceed the baseline. The report should present this result accurately.

## 10. Image, ZIP and Video Testing

The Streamlit application supports:

- Single or multiple JPG, JPEG and PNG images
- ZIP batch inference
- Per-file detection counts, model, module, technique and experiment information
- Downloadable inference summaries
- Short MP4, MOV, AVI and MPEG4 videos with frame-by-frame detection

Generate HRIPCB videos for the six defect classes:

```bash
python3 scripts/create_video_test_samples.py
open runs/video_test_samples
```

The script creates:

- `mouse_bite_only_test.mp4` — repeated Mouse_bite examples
- `six_defects_one_each_test.mp4` — one example from each class
- `six_defects_mixed_test.mp4` — mixed examples from all six classes

If the browser cannot play the output video, install `ffmpeg`. The dashboard prefers H.264 output and displays a warning when browser-compatible encoding is unavailable.

## 11. Testing and Verification

```bash
pytest -q
python3 -m compileall -q src scripts tests
python3 scripts/validate_dataset.py --root HRIPCB_UPDATE
git diff --check
```

After changing the dashboard, start Streamlit and check `http://localhost:8501`. Important checks include:

- Module filtering does not incorrectly show every technique for every module.
- The baseline control is not counted as a fifth member module.
- Model, technique, split and sorting filters work together.
- Missing checkpoints, invalid files, empty detections and video-encoding failures show readable messages.
- JSON, PDF and CSV exports do not contain `Infinity` or other non-serializable values.

## 12. Git Cleanup Policy

The following are intended to remain in Git:

- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `README.md`
- Requirements and project configuration files
- Necessary baseline metadata and the already-tracked shared checkpoint

The following are intentionally excluded from future Git commits:

- HRIPCB image splits
- New `*.pt` model weights
- New graphs, previews, batch images, videos, PDFs, sweep variants and copied datasets under `runs/`
- `.pytest_cache/`, virtual environments, temporary files and machine-specific artifacts

The current cleanup was performed locally. No commit or push was executed.
