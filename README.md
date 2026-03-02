# DroneAI -- Anomaly Detection System

**Drone-as-First-Responder** anomaly detection: YOLOv26 object detection + GRU temporal classification.

---

## Architecture

```
Video/Frames --> YOLOv26n (Detection + Embedding) --> GRU (Anomaly Classification)
                    Feature Vectors (T, D)              Probability [0..1]
```

## Project Structure

```
droneai/
├── src/
│   ├── train_detector.py      # Stage 1: Fine-tune YOLOv26n on VisDrone
│   ├── extract_embeddings.py  # Stage 2: YOLO embeddings from UCF-Crime
│   ├── train_anomaly.py       # Stage 3: GRU anomaly classifier
│   └── app.py                 # Gradio WebUI
├── datasets/
│   ├── visdrone/              # VisDrone2019-DET
│   └── ufc-crime/             # UCF-Crime (pre-extracted frames)
├── features/                  # Generated .npy embeddings
├── models/                    # Generated GRU checkpoints
├── runs/                      # Generated YOLO training runs
└── pyproject.toml
```

## Quick Start

```bash
# Install
git clone https://github.com/clitic/droneai && cd droneai
uv sync

# Run pipeline (in order)
uv run python src/train_detector.py        # Stage 1 - auto-resumes if interrupted
uv run python src/extract_embeddings.py    # Stage 2 - auto-discovers model
uv run python src/train_anomaly.py         # Stage 3

# Launch UI
uv run python src/app.py                   # http://localhost:7860
```

No CLI flags needed -- all scripts use best defaults with auto-resume.

## Pipeline

| Stage | Script | What it does |
|-------|--------|-------------|
| 1 | `train_detector.py` | YOLOv26n on VisDrone, 50 epochs, cache+AMP for speed, auto-resume |
| 2 | `extract_embeddings.py` | `model.embed()` on UCF-Crime frames, saves .npy per clip |
| 3 | `train_anomaly.py` | Bi-GRU + attention, weighted BCE, cosine LR, early stopping |
| UI | `app.py` | Video/Image/Batch analysis via Gradio |

## Datasets

See [`datasets/README.md`](datasets/README.md) for download instructions.

| Dataset | Purpose | Classes |
|---------|---------|---------|
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | Drone object detection | 10 |
| [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) | Anomaly classification | 13 + Normal |

## Tech Stack

Ultralytics YOLOv26n -- PyTorch 2.10+ -- Gradio 6.8+ -- scikit-learn -- uv
