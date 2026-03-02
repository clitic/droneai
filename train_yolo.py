"""
Stage 1 — Fine-tune YOLOv26n on the VisDrone dataset.

Usage:
    uv run python train_yolo.py --epochs 50 --batch 16
    uv run python train_yolo.py --help
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv26n on VisDrone for drone object detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Base YOLO model to fine-tune",
    )
    p.add_argument(
        "--data",
        type=str,
        default="datasets/visdrone/visdrone.yaml",
        help="Path to VisDrone dataset YAML config",
    )
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to train on (0, 1, cpu, etc.)",
    )
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    p.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for saving results",
    )
    p.add_argument("--name", type=str, default="train", help="Experiment name")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    # Hyperparameters
    p.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--lrf", type=float, default=0.01, help="Final LR (lr0 * lrf)")
    p.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    p.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    p.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs")
    p.add_argument(
        "--cos-lr", action="store_true", help="Use cosine LR scheduler"
    )
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    # Augmentation
    p.add_argument(
        "--augment", action="store_true", default=True, help="Use augmentation"
    )
    p.add_argument("--hsv-h", type=float, default=0.015, help="HSV-Hue augmentation")
    p.add_argument(
        "--hsv-s", type=float, default=0.7, help="HSV-Saturation augmentation"
    )
    p.add_argument("--hsv-v", type=float, default=0.4, help="HSV-Value augmentation")
    p.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation")
    p.add_argument("--mixup", type=float, default=0.0, help="Mixup augmentation")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate dataset config exists
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}\n"
            "Make sure datasets/visdrone/visdrone.yaml exists."
        )

    # Resolve model path for resume
    if args.resume:
        last_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if last_ckpt.exists():
            model_path = str(last_ckpt)
            print(f"  -> Resuming from: {last_ckpt}")
        else:
            raise FileNotFoundError(
                f"Cannot resume — checkpoint not found: {last_ckpt}\n"
                "Run a fresh training first, or check --project / --name."
            )
    else:
        model_path = args.model

    print("=" * 60)
    print("  DroneAI — Stage 1: YOLOv26n Fine-Tuning on VisDrone")
    print("=" * 60)
    print(f"  Model:   {model_path}")
    print(f"  Data:    {args.data}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  ImgSz:   {args.imgsz}")
    print(f"  Batch:   {args.batch}")
    print(f"  Device:  {args.device}")
    print(f"  Resume:  {args.resume}")
    print("=" * 60)

    # Load model
    model = YOLO(model_path)

    # Train
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        # Learning rate
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        patience=args.patience,
        # Augmentation
        augment=args.augment,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        mixup=args.mixup,
        # VisDrone-specific: many small objects, enable multi-scale
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    # Validate best model
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n[OK] Training complete! Best weights: {best_weights}")
        print("Running validation on best model...")
        best_model = YOLO(str(best_weights))
        metrics = best_model.val(data=str(data_path), imgsz=args.imgsz)
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
    else:
        print(f"\n[WARN] Best weights not found at {best_weights}")

    print("\n>> Next step: extract features with extract_features.py")


if __name__ == "__main__":
    main()
