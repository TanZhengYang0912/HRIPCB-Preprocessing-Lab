# Shared YOLOv8s PCB Defect Detection Baseline

## Goal

Build one reproducible YOLOv8s object-detection baseline that all four project members can share for later denoising and contrast-enhancement experiments.

## Confirmed dataset

The local dataset is `HRIPCB_UPDATE` with the existing split:

- `train`: 485 JPEG images and 485 YOLO label files
- `val`: 138 JPEG images and 138 YOLO label files
- `test`: 70 JPEG images and 70 YOLO label files
- six classes with IDs `0` through `5`

The existing split will be preserved for the first baseline. The current filenames show the same PCB prefixes across multiple splits, so this baseline must be described as the supplied split rather than as a verified template-board group split.

## Architecture

The detector will be a single pretrained YOLOv8s model fine-tuned on the clean training images. No synthetic noise, low-contrast degradation, denoising, or contrast enhancement will be used during baseline training. The resulting `best.pt` checkpoint will be fixed and reused by all later preprocessing experiments.

The local YOLO dataset configuration will use paths relative to the project root. Training, validation, and test evaluation will use the same six class names and the same image size, confidence threshold, random seed, and model checkpoint.

## Training policy

- Model: YOLOv8s pretrained weights
- Classes: `Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spurious_copper`, `Spur`
- Image size: 1024 pixels
- Seed: 42
- Maximum epochs: 100
- Early-stopping patience: 20
- Initial batch size: 4, reduced only if the local MPS memory requires it
- Device: Apple Silicon MPS when available
- Data loader workers: 0 for stable local macOS execution
- Baseline images remain clean; degradation is reserved for later inference experiments

## Components

1. Dataset validation checks image/label pairing, readable files, class IDs, normalized bounding boxes, and split counts.
2. Dataset configuration resolves local paths and class names.
3. Training entry point runs the shared YOLOv8s training with explicit settings.
4. Evaluation entry point evaluates the fixed checkpoint on clean validation and test sets.
5. Results are stored with the checkpoint, run arguments, metrics, and environment information.

## Validation and acceptance criteria

The baseline is accepted when:

1. Dataset validation passes for all 693 images and labels.
2. A one-epoch smoke training run completes successfully on the local machine.
3. Full training completes or stops through the configured early-stopping rule.
4. A `best.pt` checkpoint exists and can be loaded for inference.
5. Clean validation and test evaluation produce detection metrics and per-class results.
6. The run records the exact configuration needed for every member to reuse the same model.

## Known limitation

The supplied `data.yaml` contains Kaggle paths and must be replaced or supplemented with local paths. The existing train/validation/test split has not yet been proven to be template-board grouped. That limitation will be reported and addressed separately if a reliable template mapping is found.
