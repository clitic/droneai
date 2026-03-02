# Datasets

This directory holds the two datasets used by the DroneAI pipeline. **They are gitignored** — you must download them yourself.

---

## VisDrone2019-DET

**Purpose:** Fine-tune YOLOv26n for drone-perspective object detection (Stage 1).

**Download:** [VisDrone-Dataset GitHub](https://github.com/VisDrone/VisDrone-Dataset)

Place the data so the structure matches:

```
datasets/visdrone/
├── visdrone.yaml                      # Config (already included)
├── VisDrone2019-DET-train/
│   ├── images/                        # 6,471 training images
│   └── annotations/
├── VisDrone2019-DET-val/
│   ├── images/                        # 548 validation images
│   └── annotations/
└── VisDrone2019-DET-test-dev/
    ├── images/                        # 1,610 test images
    └── annotations/
```

**10 classes:** pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.

> [!NOTE]
> `visdrone.yaml` is pre-configured and tracked in git. The image/annotation folders are gitignored.

---

## UCF-Crime

**Purpose:** Extract temporal feature embeddings for anomaly classification (Stage 2).

**Download:** [UCF-Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/)

This project expects the dataset as **pre-extracted PNG frames** (not raw videos):

```
datasets/ufc-crime/
├── Train/
│   ├── Abuse/                # Abuse001_x264_10.png, ...
│   ├── Arrest/
│   ├── Arson/
│   ├── Assault/
│   ├── Burglary/
│   ├── Explosion/
│   ├── Fighting/
│   ├── NormalVideos/         # Label = 0 (Normal)
│   ├── RoadAccidents/
│   ├── Robbery/
│   ├── Shooting/
│   ├── Shoplifting/
│   ├── Stealing/
│   └── Vandalism/
└── Test/
    └── (same 14 categories)
```

**13 anomaly categories + 1 Normal.** For binary classification, all crime categories are collapsed into a single "Anomaly" label (1), while `NormalVideos` is label (0).

### Frame naming convention

Frames follow the pattern `<ClipName>_<FrameNumber>.png`, e.g.:
- `Abuse001_x264_1580.png` → clip `Abuse001_x264`, frame 1580

The `extract_features.py` script groups frames by clip name automatically.

> [!TIP]
> If you have the raw `.mp4` videos instead, extract frames with ffmpeg:
> ```bash
> ffmpeg -i video.mp4 -vf "fps=1" output_dir/video_%04d.png
> ```
