"""
Train a YOLO26 model on the unified drone dataset.

Usage:
    python train_yolo.py [--model yolo26n.pt] [--epochs 150] [--imgsz 640] [--batch -1] [--device 0]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 on unified drone dataset")
    parser.add_argument("--model", type=str, default="yolo26n.pt",
                        help="Pretrained model to start from (e.g. yolo26n.pt, yolo26s.pt, yolo26m.pt)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (640 recommended for small objects)")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto)")
    parser.add_argument("--device", type=str, default="0", help="Device: 0 for GPU, cpu for CPU")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--name", type=str, default="drone_detect", help="Run name for saving results")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Resolve data.yaml with whitespace-safe path
    data_yaml = str(Path(__file__).resolve().parent / "data" / "unified" / "data.yaml")

    print(f"📦  Model     : {args.model}")
    print(f"📁  Dataset   : {data_yaml}")
    print(f"🔄  Epochs    : {args.epochs}")
    print(f"📐  Image size: {args.imgsz}")
    print(f"📊  Batch size: {args.batch}")
    print(f"🖥️  Device    : {args.device}")
    print()

    model = YOLO(args.model)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project=str(Path(__file__).resolve().parent / "runs"),
        exist_ok=True,
        resume=args.resume,
        # ── Optimizer ──
        optimizer="auto",          # lets YOLO26 use its MuSGD optimizer
        lr0=0.01,                  # initial learning rate
        lrf=0.01,                  # final LR = lr0 * lrf (cosine decay)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,               # cosine LR scheduler for smoother convergence
        # ── Augmentation (tuned for aerial/drone views) ──
        hsv_h=0.015,               # hue shift
        hsv_s=0.7,                 # saturation shift
        hsv_v=0.4,                 # brightness shift
        degrees=10.0,              # rotation — drone views are orientation-agnostic
        translate=0.2,             # translation
        scale=0.5,                 # multi-scale for varying altitudes
        flipud=0.5,                # vertical flip — top-down views have no gravity bias
        fliplr=0.5,                # horizontal flip
        mosaic=1.0,                # mosaic augmentation — critical for small objects
        mixup=0.1,                 # mixup — regularization
        copy_paste=0.1,            # copy-paste augmentation — boosts small object detection
        # ── Training strategy ──
        patience=15,               # early stopping — nano converges fast
        close_mosaic=10,           # disable mosaic in last 10 epochs for fine-tuning
        amp=True,                  # mixed precision — faster on RTX 3050
        cache="disk",              # cache images to disk for faster loading after first epoch
        verbose=True,
    )

    print("\n✅  Training complete!")
    print(f"   Results saved to: {Path(__file__).resolve().parent / 'runs' / args.name}")


if __name__ == "__main__":
    main()
