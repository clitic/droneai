from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    data = Path("datasets/visdrone/visdrone.yaml")
    if not data.exists():
        raise FileNotFoundError(f"Dataset config not found: {data}")

    last = Path("runs/detect/visdrone/weights/last.pt")
    if last.exists():
        model_path, resume = str(last), True
    else:
        model_path, resume = "yolo26n.pt", False

    model = YOLO(model_path)
    model.train(
        data=str(data),
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",
        workers=8,
        name="visdrone",
        exist_ok=True,
        resume=resume,
        cache="disk",
        rect=False,
        close_mosaic=10,
        amp=True,
        save_period=-1,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        patience=20,
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


if __name__ == "__main__":
    main()
