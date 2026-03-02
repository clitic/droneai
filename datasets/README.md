# Datasets

Both datasets are gitignored -- download them yourself.

---

## VisDrone2019-DET

Fine-tune YOLOv26n for drone-perspective detection (Stage 1).

**Download:** [VisDrone-Dataset GitHub](https://github.com/VisDrone/VisDrone-Dataset)

```
datasets/visdrone/
├── visdrone.yaml                      # Config (tracked in git)
├── VisDrone2019-DET-train/images/     # 6,471 images
├── VisDrone2019-DET-val/images/       # 548 images
└── VisDrone2019-DET-test-dev/images/  # 1,610 images
```

10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.

---

## UCF-Crime

Extract temporal embeddings for anomaly classification (Stage 2).

**Download:** [UCF-Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/)

Expected as pre-extracted PNG frames:

```
datasets/ufc-crime/
├── Train/
│   ├── Abuse/            # Abuse001_x264_10.png, Abuse001_x264_20.png, ...
│   ├── Arrest/
│   ├── ...
│   ├── NormalVideos/     # Label 0 (Normal)
│   └── Vandalism/
└── Test/
    └── (same 14 categories)
```

13 anomaly categories + NormalVideos. All crime types collapse to label 1 (Anomaly).

Frame naming: `<ClipName>_<FrameNumber>.png` -- the extraction script groups by clip automatically.

If you have raw videos, extract frames with:
```bash
ffmpeg -i video.mp4 -vf "fps=1" output_dir/video_%04d.png
```
