# 🚁 DroneAI — Anomaly Detection System

**Drone-as-First-Responder** anomaly detection using a two-stage pipeline: YOLOv26 object detection + GRU temporal classification.

![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.10%2B-ee4c2c)
![Ultralytics](https://img.shields.io/badge/ultralytics-8.4%2B-00FFFF)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Architecture

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  Video Feed  │ ──▶ │  YOLOv26n (Stage 1) │ ──▶ │  GRU Classifier      │
│  / Frames    │     │  Object Detection   │     │  (Stage 3)           │
└──────────────┘     │  + Embedding        │     │  Temporal Anomaly    │
                     │  Extraction         │     │  Classification      │
                     └─────────────────────┘     └──────────────────────┘
                            │                           │
                     Feature Vectors               Anomaly Prob.
                        (T, D)                     [0.0 → 1.0]
```

**Stage 1** fine-tunes YOLOv26n on drone-perspective imagery (VisDrone) for robust aerial object detection. **Stage 2** extracts penultimate-layer embeddings from UCF-Crime video frames. **Stage 3** trains a bidirectional GRU with attention pooling to classify temporal sequences as Normal or Anomaly.

---

## Project Structure

```
droneai/
├── datasets/
│   ├── visdrone/              # VisDrone2019-DET dataset
│   │   └── visdrone.yaml      # Dataset config
│   └── ufc-crime/             # UCF-Crime (pre-extracted frames)
│       ├── Train/             # 14 category folders
│       └── Test/              # 14 category folders
├── train_yolo.py              # Stage 1: Fine-tune YOLOv26n
├── extract_features.py        # Stage 2: Embed frames → .npy
├── train_classifier.py        # Stage 3: GRU temporal classifier
├── inference.py               # CLI end-to-end inference
├── app.py                     # Gradio WebUI
├── features/                  # Generated: .npy embeddings
├── models/                    # Generated: GRU checkpoints
├── runs/                      # Generated: YOLO training runs
└── pyproject.toml
```

---

## Quick Start

### Prerequisites

- Python 3.14+
- CUDA-capable GPU (tested on RTX 3050)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone <repo-url> droneai
cd droneai
uv sync
```

### Run the Pipeline

Execute each stage **in order**:

```bash
# Stage 1 — Fine-tune YOLOv26n on VisDrone
uv run python train_yolo.py --epochs 50 --batch 16

# Stage 2 — Extract embeddings from UCF-Crime frames
uv run python extract_features.py --batch-size 32

# Stage 3 — Train GRU anomaly classifier
uv run python train_classifier.py --epochs 30 --hidden-size 128
```

### Inference

**Gradio WebUI** (recommended):
```bash
uv run python app.py
# Open http://localhost:7860
```

**CLI**:
```bash
uv run python inference.py --source path/to/video.mp4 --save-video
```

---

## Pipeline Details

### Stage 1 — YOLO Fine-Tuning (`train_yolo.py`)

Fine-tunes YOLOv26n (nano) on VisDrone2019-DET for drone-perspective object detection across 10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Training epochs |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Input image size |
| `--device` | 0 | GPU device |
| `--cos-lr` | off | Cosine LR scheduler |
| `--patience` | 20 | Early stopping patience |

### Stage 2 — Feature Extraction (`extract_features.py`)

Runs `model.embed()` on UCF-Crime frames to extract penultimate-layer feature vectors. Frames are grouped by video clip, and each clip's features are saved as a `(T, D)` numpy array.

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `runs/detect/train/weights/best.pt` | YOLO weights |
| `--batch-size` | 16 | Inference batch size |
| `--max-frames` | 0 (all) | Cap frames per clip |

**Output:** `features/{Train,Test}/<Category>/<clip>.npy` + `features/manifest.json`

### Stage 3 — GRU Classifier (`train_classifier.py`)

Bidirectional GRU with learned attention pooling over time steps. Trained with class-weighted BCE loss (auto-balances Normal vs. Anomaly) and cosine-annealing LR.

| Flag | Default | Description |
|------|---------|-------------|
| `--hidden-size` | 128 | GRU hidden size |
| `--num-layers` | 2 | GRU depth |
| `--seq-len` | 64 | Fixed sequence length |
| `--lr` | 1e-4 | Learning rate |
| `--patience` | 10 | Early stopping patience |

**Output:** `models/gru_best.pt`

### Gradio WebUI (`app.py`)

| Tab | Function |
|-----|----------|
| 🎬 **Video Analysis** | Upload video → YOLO detection + GRU anomaly scoring |
| 🖼️ **Image Detection** | Single-image YOLO detection (no temporal analysis) |
| 📊 **Batch Process** | Analyze a folder of videos/clips |
| ℹ️ **About** | Pipeline docs and quick-start guide |

---

## Datasets

See [`datasets/README.md`](datasets/README.md) for download instructions and structure details.

| Dataset | Purpose | Classes |
|---------|---------|---------|
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | Drone object detection | 10 (pedestrian, car, bus, …) |
| [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) | Anomaly classification | 13 crime types + Normal |

---

## Tech Stack

- **Detection**: [Ultralytics](https://docs.ultralytics.com/) YOLOv26n
- **Deep Learning**: [PyTorch](https://pytorch.org/) 2.10+ (CUDA 13.0)
- **UI**: [Gradio](https://gradio.app/) 6.8+
- **Env**: [uv](https://docs.astral.sh/uv/) for dependency management
- **Metrics**: scikit-learn (AUC-ROC, classification report)

---

## License

MIT
