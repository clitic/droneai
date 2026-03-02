"""
Stage 1 -- Fine-tune YOLOv26n on VisDrone.
Auto-resumes from last checkpoint if available.

Usage:
    uv run python src/train_detector.py
"""

from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    data = Path("datasets/visdrone/visdrone.yaml")
    if not data.exists():
        raise FileNotFoundError(f"Dataset config not found: {data}")

    # Auto-resume from last checkpoint if previous run exists
    last = Path("runs/visdrone/weights/last.pt")
    if last.exists():
        model_path, resume = str(last), True
        print(f"  -> Resuming from: {last}")
    else:
        model_path, resume = "yolo26n.pt", False

    print("=" * 60)
    print("  DroneAI -- Stage 1: YOLOv26n on VisDrone")
    print("=" * 60)
    print(f"  Model:  {model_path}")
    print(f"  Resume: {resume}")
    print("=" * 60)

    model = YOLO(model_path)
    model.train(
        data=str(data),
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",
        workers=8,
        project="runs",
        name="visdrone",
        exist_ok=True,
        resume=resume,
        # -- Speed optimizations --
        cache=True,             # cache images in RAM for faster loading
        rect=False,             # rectangular training (can hurt mAP on small objects)
        close_mosaic=10,        # disable mosaic in last 10 epochs
        amp=True,               # automatic mixed precision (FP16)
        save_period=-1,         # only save best + last (skip periodic saves)
        # -- Learning rate --
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        patience=20,
        # -- Augmentation --
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.0,
        save=True,
        val=True,
        plots=True,
        verbose=True,
    )

    best = Path("runs/visdrone/weights/best.pt")
    if best.exists():
        print(f"\n[OK] Training complete! Best weights: {best}")
        print("Running validation...")
        metrics = YOLO(str(best)).val(data=str(data), imgsz=640)
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
    else:
        print(f"\n[WARN] Best weights not found at {best}")

    print("\n>> Next: uv run python src/extract_embeddings.py")


if __name__ == "__main__":
    main()
