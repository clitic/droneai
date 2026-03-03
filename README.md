# DroneAI -- Anomaly Detection System

**Drone-as-First-Responder** surveillance system that detects objects in drone footage and classifies anomalous activities using deep learning.

---

## Architecture

The system uses a two-stage approach: a vision model extracts spatial features from each frame, then a temporal model analyzes how those features change over time to classify the type of activity.

```
                    Stage 1                    Stage 2                    Stage 3
                 (trained once)             (run once)               (trained once)
                      |                        |                          |
                      v                        v                          v
Video Frames --> YOLOv26n Backbone --> Feature Embeddings --> Bi-GRU + Attention --> Class Prediction
                 (per-frame)            (T, 576)              (temporal)           14 classes
```

### YOLOv26n -- Object Detector (Stage 1)

- **Model:** YOLOv26n (nano variant, ~2.6M params)
- **Training data:** [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) -- 6,471 drone-perspective images, 10 object classes (pedestrian, car, bicycle, truck, bus, etc.)
- **Purpose:** Fine-tuned to detect objects commonly seen from drones. The backbone's penultimate layer produces a 576-dimensional embedding per frame that encodes *what objects are present and where*.
- **Training:** 50 epochs, 640px images, disk caching, AMP (FP16), cosine LR, auto-resume from checkpoint

### Bi-GRU + Attention -- Anomaly Classifier (Stage 3)

- **Model:** 3-layer Bidirectional GRU with learned attention pooling (~1M params)
- **Training data:** [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) -- 1,610 clips across 14 categories
- **Input:** Sequence of YOLO embeddings (64 frames x 576 dims) from a video clip
- **Output:** One of 14 classes:
  - `Normal` -- no anomaly
  - `Abuse`, `Arrest`, `Arson`, `Assault`, `Burglary`, `Explosion`, `Fighting`, `RoadAccidents`, `Robbery`, `Shooting`, `Shoplifting`, `Stealing`, `Vandalism`
- **How it works:**
  1. The GRU reads the embedding sequence bidirectionally, capturing temporal patterns in both directions
  2. An attention layer learns which frames are most important for classification
  3. A classification head (LayerNorm → GELU → Linear) maps the attended representation to one of 14 classes
- **Training:** CrossEntropyLoss with label smoothing (0.1) and inverse-frequency class weights, AdamW (lr=3e-4), cosine annealing, early stopping (patience=20), 100 epochs

### Inference Pipeline (Gradio UI)

```
Upload Video --> Read Frames --> YOLO embed() --> Bi-GRU --> Top-5 Class Probabilities
Upload Image --> YOLO detect() --> Annotated Image (bounding boxes)
```

---

## Project Structure

```
droneai/
├── src/
│   ├── train_yolo.py   # Stage 1: Fine-tune YOLOv26n on VisDrone
│   ├── embed.py        # Stage 2: Extract YOLO embeddings from UCF-Crime
│   ├── train_gru.py    # Stage 3: Train Bi-GRU anomaly classifier
│   └── app.py          # Gradio WebUI (video + image analysis)
├── datasets/
│   ├── visdrone/              # VisDrone2019-DET dataset
│   ├── ufc-crime/             # UCF-Crime frames (64x64 PNG)
│   └── ucf-crime-features/    # Generated .npy embeddings
├── runs/                      # YOLO training runs + GRU checkpoint
└── pyproject.toml
```

## Quick Start

```bash
# Install
git clone https://github.com/clitic/droneai && cd droneai
uv sync

# Run pipeline (in order)
uv run python src/train_yolo.py   # Stage 1 - auto-resumes if interrupted
uv run python src/embed.py        # Stage 2
uv run python src/train_gru.py    # Stage 3

# Launch UI
uv run python src/app.py          # http://localhost:7860
```

## Pipeline Details

| Stage | Script | Model | Data | Output |
|-------|--------|-------|------|--------|
| 1 | `train_yolo.py` | YOLOv26n | VisDrone (6.4K images) | `runs/detect/visdrone/weights/best.pt` |
| 2 | `embed.py` | Trained YOLO | UCF-Crime (1.26M frames) | `datasets/ucf-crime-features/*.npy` |
| 3 | `train_gru.py` | Bi-GRU (14-class) | Extracted embeddings | `runs/gru_best.pt` |
| UI | `app.py` | Both models | User uploads | Anomaly class + detection |

## Datasets

See [`datasets/README.md`](datasets/README.md) for download instructions.

| Dataset | Purpose | Size | Classes |
|---------|---------|------|---------|
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | Object detection from drones | 8.6K images | 10 (pedestrian, car, ...) |
| [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) | Anomaly classification | 1.26M frames, 64x64 | 14 (Normal + 13 anomaly types) |

## Tech Stack

Ultralytics YOLOv26n -- PyTorch 2.10+ -- Gradio 6.8+ -- scikit-learn -- uv
