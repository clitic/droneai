# Datasets

Both datasets are gitignored -- download them yourself.

---

## VisDrone2019-DET

Used to fine-tune YOLOv26n for drone-perspective object detection (Stage 1). The trained model's backbone provides the feature embeddings used by the anomaly classifier.

**Download:** [VisDrone-Dataset GitHub](https://github.com/VisDrone/VisDrone-Dataset)

```
datasets/visdrone/
├── visdrone.yaml                      # Config (tracked in git)
├── VisDrone2019-DET-train/images/     # 6,471 images
├── VisDrone2019-DET-val/images/       # 548 images
└── VisDrone2019-DET-test-dev/images/  # 1,610 images
```

**10 classes:** pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

---

## UCF-Crime

Used for training the temporal anomaly classifier (Stages 2-3). Pre-extracted video frames are passed through YOLO to generate embeddings, which the Bi-GRU then learns to classify.

**Download:** [UCF-Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/)

Expected as pre-extracted PNG frames:

```
datasets/ufc-crime/
├── Train/
│   ├── Abuse/            # Abuse001_x264_10.png, Abuse001_x264_20.png, ...
│   ├── Arrest/
│   ├── Arson/
│   ├── Assault/
│   ├── Burglary/
│   ├── Explosion/
│   ├── Fighting/
│   ├── NormalVideos/     # Normal activity (no anomaly)
│   ├── RoadAccidents/
│   ├── Robbery/
│   ├── Shooting/
│   ├── Shoplifting/
│   ├── Stealing/
│   └── Vandalism/
└── Test/
    └── (same 14 categories)
```

**14 classes:** Each folder becomes a class for the multi-class GRU classifier. NormalVideos = class 0 (no anomaly), all others are specific anomaly types.

Frame naming: `<ClipName>_<FrameNumber>.png` -- the extraction script groups frames by clip name automatically.

If you have raw videos, extract frames with:
```bash
ffmpeg -i video.mp4 -vf "fps=1" output_dir/video_%04d.png
```
