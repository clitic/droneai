# Datasets

```bash
curl -L https://www.kaggle.com/api/v1/datasets/download/kushagrapandya/visdrone-dataset -o visdrone.zip
curl -L https://www.kaggle.com/api/v1/datasets/download/odins0n/ucf-crime-dataset -o ucf-crime.zip
```

Configure path in `visdrone.yaml`: `path: datasets/visdrone`

## VisDrone2019-DET

Fine-tunes YOLOv26n for drone-perspective object detection (`train_yolo.py`).

```
datasets/visdrone/
├── visdrone.yaml                      # Config (tracked in git)
├── VisDrone2019-DET-train/images/     # 6,471 images
├── VisDrone2019-DET-val/images/       # 548 images
└── VisDrone2019-DET-test-dev/images/  # 1,610 images
```

**10 classes:** pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

## UCF-Crime

Pre-extracted 64x64 PNG frames from UCF-Crime videos. Every 10th frame is extracted.

Used by `embed.py` (Stage 2) to generate YOLO embeddings, then by `train_gru.py` (Stage 3) to train the anomaly classifier.

```
datasets/ufc-crime/
├── Train/                  # 1,266,345 frames
│   ├── Abuse/
│   ├── Arrest/
│   ├── Arson/
│   ├── Assault/
│   ├── Burglary/
│   ├── Explosion/
│   ├── Fighting/
│   ├── NormalVideos/
│   ├── RoadAccidents/
│   ├── Robbery/
│   ├── Shooting/
│   ├── Shoplifting/
│   ├── Stealing/
│   └── Vandalism/
└── Test/                   # 111,308 frames
    └── (same 14 categories)
```

**14 classes:** NormalVideos = class 0, all others are anomaly types.

Frame naming: `<ClipName>_<FrameNumber>.png` (e.g. `Abuse001_x264_10.png`)