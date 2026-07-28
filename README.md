# PCB Defect Detection — YOLOv8s Shared Baseline

Object-detection baseline for the Image Processing assignment.  
Detects **6 PCB defect classes** from the HRIPCB dataset using a fine-tuned YOLOv8s model.

---

## 📋 Project Structure

```
ImageProcessing-Assignment/
├── configs/              # YAML configs for training
│   ├── baseline.yaml
│   └── hripcb_local.yaml
├── scripts/              # Training, evaluation & validation scripts
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   └── validate_dataset.py
├── src/
│   └── hripcb_baseline/  # Python package (dataset loader, config)
├── runs/
│   ├── baseline/
│   │   ├── weights/
│   │   │   └── best.pt   ← ✅ Shared checkpoint (all members use this)
│   │   ├── results.csv
│   │   └── *.png         # Training curves
│   └── evaluation/       # val / test metrics & curves
│       ├── val/metrics.json
│       └── test/metrics.json
├── artifacts/
│   └── baseline-manifest.json
├── HRIPCB_UPDATE/
│   └── data.yaml         # Dataset YAML (images NOT in repo — see below)
├── requirements.txt
└── pyproject.toml
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/TanZhengYang0912/Image_Processing.git
cd Image_Processing
```

### 2. Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 3. Download the dataset

The dataset images are **not tracked in Git** (too large).  
Place the HRIPCB dataset under `HRIPCB_UPDATE/` so the folder looks like:

```
HRIPCB_UPDATE/
├── data.yaml
├── train/
│   ├── images/   (485 images)
│   └── labels/
├── val/
│   ├── images/   (138 images)
│   └── labels/
└── test/
    ├── images/   (70 images)
    └── labels/
```

Verify the dataset structure:

```bash
python3 scripts/validate_dataset.py --root HRIPCB_UPDATE
```

---

## 📊 Dataset

| Split | Images | Classes |
|-------|-------:|---------|
| train | 485    | Missing_hole, Mouse_bite, Open_circuit, Short, Spurious_copper, Spur |
| val   | 138    | same |
| test  | 70     | same |

---

## 🤖 Shared Model Checkpoint

The trained checkpoint every member must reuse is:

```
runs/baseline/weights/best.pt
```

> ⚠️ **Do NOT retrain the baseline.** Only test-time preprocessing may vary across experiments.

---

## 🏋️ Training (for reference only)

The baseline was trained with:

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

**Key hyperparameters:** `imgsz=1024`, `seed=42`, `patience=20`, `batch=4`, Apple MPS when available.  
The formal run trained for **72 epochs** and stopped at patience 20; best checkpoint saved at **epoch 52**.

---

## 📈 Baseline Results

Evaluated with `imgsz=1024`, `conf=0.25`, `iou=0.7`:

| Split | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|-------:|----------:|----------:|-------:|------:|---------:|
| val   | 138    | 600       | 0.972     | 0.944  | 0.942 | 0.515    |
| test  | 70     | 293       | 0.952     | 0.923  | 0.921 | 0.489    |

Machine-readable metrics: `runs/evaluation/val/metrics.json` and `runs/evaluation/test/metrics.json`.

---

## 🧪 Evaluation

Run evaluation against the shared checkpoint:

```bash
# Validation split
python3 scripts/evaluate_baseline.py \
  --weights runs/baseline/weights/best.pt \
  --data configs/hripcb_local.yaml \
  --split val

# Test split
python3 scripts/evaluate_baseline.py \
  --weights runs/baseline/weights/best.pt \
  --data configs/hripcb_local.yaml \
  --split test
```

## 🔬 Member 3: Bilateral Filtering + AGCWD

### Preliminary noise study

Member 3's Mode A experiment keeps the shared `best.pt` detector frozen. It
adds reproducible Gaussian noise to the Y channel, applies Bilateral Filtering
and AGCWD to Y only, and evaluates the same detector on clean, noisy, ablated,
and combined inputs. Validation selects one global parameter combination;
the test split is used only for the final comparison.

```bash
python3 scripts/run_member3.py \
  --dataset-root /path/to/HRIPCB_UPDATE \
  --weights runs/baseline/weights/best.pt \
  --output runs/member3
```

The experiment tests noise levels `σ=10, 25, 40`, uses fixed seeds `42, 43,
44`, and writes `summary.json`, `metrics.json`, and `comparison.csv` under
`runs/member3/`.

### Formal validation study

The team-agreed formal Member 3 study uses validation data only. It compares
16 clean-image preprocessing conditions with the frozen baseline checkpoint:
Original, three Bilateral presets, three AGCWD plus gamma presets, and nine
combined presets. It selects the Member 3 candidate by mAP50-95 and writes
results under `runs/member3_formal/`; it does not use or rank test data.

```bash
python3 scripts/run_member3_formal.py \
  --dataset-root /path/to/HRIPCB_UPDATE \
  --weights runs/baseline/weights/best.pt \
  --output runs/member3_formal
```

The fixed contract is `val` (138 images), `imgsz=1024`, `conf=0.25`,
`iou=0.70`, `workers=0`, and automatic MPS/CPU device selection. Bilateral
presets are `(5,25,25)`, `(7,50,50)`, and `(9,75,75)`; gamma presets are
`0.8`, `1.0`, and `1.2`. The implementation keeps AGCWD alpha fixed at `0.75`
and applies the selected global gamma after AGCWD.

### Interactive Member 3 Demo

Install the dashboard dependency and launch the local Streamlit app:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run scripts/member3_demo.py
```

The dashboard accepts one JPG, JPEG, or PNG PCB image and shows the original
image, the selected formal preprocessing result, and YOLOv8s detection boxes.
It supports Original, Bilateral Filtering, AGCWD plus gamma, and Bilateral
Filtering plus AGCWD plus gamma with the formal presets. The detector remains
the frozen `runs/baseline/weights/best.pt` checkpoint.

Set the optional dataset-root field to `HRIPCB_UPDATE` (or the external
dataset path) when you upload a known validation/test image and want to see
its Ground Truth boxes. Interactive output images and metadata are saved
under `runs/member3_demo/`.

The `runs/member3_formal/comparison.csv` file is the formal validation summary:
each row records the complete settings, detection metrics, PSNR, SSIM, and
processing time for one condition. The dashboard never combines these
validation results with the preliminary `runs/member3/` test rows. Its
single-image detections are visual results and should not be interpreted as
dataset-level mAP.

---

## 🤝 Collaboration Rules

1. **Use `best.pt` as-is** — do not modify the checkpoint or retrain.
2. **Only vary test-time image preprocessing** in your experiments.
3. **Keep class order, image size, confidence threshold, and dataset split unchanged.**
4. **Do not commit dataset images** — add your own experiments to `runs/` subdirectories.
5. **Do not commit `*.pt` files** other than `runs/baseline/weights/best.pt`.

---

## ⚠️ Known Limitations

- The train/val/test split was not verified as a template-board group split.  
  Filename prefixes appear across multiple splits. This limitation must be stated in the report.

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full dependency list.  
Key dependencies: `ultralytics`, `torch`, `torchvision`.
