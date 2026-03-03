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
├── visdrone.yaml
├── VisDrone2019-DET-train/images/ 
├── VisDrone2019-DET-val/images/
└── VisDrone2019-DET-test-dev/images/
```

## UCF-Crime

Pre-extracted 64x64 PNG frames from UCF-Crime videos. Every 10th frame is extracted.

Used by `embed.py` (Stage 2) to generate YOLO embeddings, then by `train_gru.py` (Stage 3) to train the anomaly classifier.

```
datasets/ufc-crime/
├── Train/
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
└── Test/
    └── (same 14 categories)
```
