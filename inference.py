"""
End-to-end inference — YOLO detection + GRU anomaly scoring.

Accepts a video file or a folder of frames, runs the full pipeline, and
outputs annotated frames with bounding boxes + anomaly probability.

Usage:
    uv run python inference.py --source path/to/video.mp4
    uv run python inference.py --source path/to/frames_folder/
    uv run python inference.py --help
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Import the GRU model class from train_classifier
from train_classifier import AnomalyGRU


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DroneAI end-to-end inference: YOLO + GRU anomaly detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a video file or folder of frame images",
    )
    p.add_argument(
        "--yolo-model",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to fine-tuned YOLO weights",
    )
    p.add_argument(
        "--gru-model",
        type=str,
        default="models/gru_best.pt",
        help="Path to trained GRU checkpoint",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save annotated frames",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Anomaly probability threshold",
    )
    p.add_argument("--imgsz", type=int, default=640, help="YOLO input image size")
    p.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cuda, cpu)"
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="GRU sequence length (must match training)",
    )
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated output as video",
    )
    return p.parse_args()


def load_frames(source: str) -> list[np.ndarray]:
    """Load frames from a video file or image directory."""
    source_path = Path(source)
    frames = []

    if source_path.is_file():
        # Video file
        cap = cv2.VideoCapture(str(source_path))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"  Loaded {len(frames)} frames from video: {source_path.name}")
    elif source_path.is_dir():
        # Directory of images
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        image_paths = sorted(
            [p for p in source_path.iterdir() if p.suffix.lower() in extensions]
        )
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frames.append(frame)
        print(f"  Loaded {len(frames)} frames from directory: {source_path.name}")
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    return frames


def extract_frame_embeddings(
    yolo_model: YOLO,
    frames: list[np.ndarray],
    batch_size: int = 16,
) -> np.ndarray:
    """Extract YOLO embeddings from frames.

    Returns:
        np.ndarray of shape (T, D).
    """
    all_embeddings = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        embeddings = yolo_model.embed(batch)

        for emb in embeddings:
            if hasattr(emb, "cpu"):
                emb_np = emb.cpu().numpy()
            else:
                emb_np = np.array(emb)
            all_embeddings.append(emb_np.flatten())

    return np.stack(all_embeddings, axis=0)


def predict_anomaly(
    gru_model: AnomalyGRU,
    embeddings: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> float:
    """Run GRU inference on embeddings and return anomaly probability."""
    T, D = embeddings.shape

    # Pad or subsample to seq_len
    if T >= seq_len:
        indices = np.linspace(0, T - 1, seq_len, dtype=int)
        features = embeddings[indices]
    else:
        pad = np.zeros((seq_len - T, D), dtype=embeddings.dtype)
        features = np.concatenate([embeddings, pad], axis=0)

    # (1, seq_len, D)
    x = torch.from_numpy(features).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = gru_model(x)
        prob = torch.sigmoid(logits).item()

    return prob


def annotate_frame(
    frame: np.ndarray,
    yolo_results,
    anomaly_prob: float,
    threshold: float,
) -> np.ndarray:
    """Draw YOLO bounding boxes and anomaly score on a frame."""
    annotated = frame.copy()

    # Draw YOLO detections
    if yolo_results and len(yolo_results) > 0:
        result = yolo_results[0]
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                cls_name = result.names.get(cls, str(cls))

                color = (0, 255, 0)  # Green for normal detections
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

    # Anomaly overlay
    is_anomaly = anomaly_prob >= threshold
    status_text = f"ANOMALY: {anomaly_prob:.1%}" if is_anomaly else f"Normal: {anomaly_prob:.1%}"
    status_color = (0, 0, 255) if is_anomaly else (0, 200, 0)  # Red or Green

    # Background bar
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(
        annotated, status_text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2,
    )

    # Probability bar
    bar_width = int((w - 20) * anomaly_prob)
    bar_color = (0, 0, 255) if is_anomaly else (0, 200, 0)
    cv2.rectangle(annotated, (10, 35), (10 + bar_width, 38), bar_color, -1)
    cv2.rectangle(annotated, (10, 35), (w - 10, 38), (100, 100, 100), 1)

    return annotated


def main() -> None:
    args = parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("  DroneAI — End-to-End Inference")
    print("=" * 60)
    print(f"  Source:    {args.source}")
    print(f"  YOLO:     {args.yolo_model}")
    print(f"  GRU:      {args.gru_model}")
    print(f"  Device:   {device}")
    print(f"  Threshold: {args.threshold}")
    print("=" * 60)

    # Load models
    yolo_model = YOLO(args.yolo_model)

    # Load GRU
    gru_ckpt = torch.load(args.gru_model, map_location=device, weights_only=True)
    gru_model = AnomalyGRU(
        input_dim=gru_ckpt["input_dim"],
        hidden_size=gru_ckpt["hidden_size"],
        num_layers=gru_ckpt["num_layers"],
        dropout=gru_ckpt["dropout"],
        bidirectional=gru_ckpt["bidirectional"],
    ).to(device)
    gru_model.load_state_dict(gru_ckpt["model_state_dict"])
    gru_model.eval()

    # Load frames
    frames = load_frames(args.source)
    if not frames:
        print("❌ No frames loaded. Check your source path.")
        return

    # Extract embeddings
    print("\n📐 Extracting embeddings...")
    embeddings = extract_frame_embeddings(yolo_model, frames)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Predict anomaly
    anomaly_prob = predict_anomaly(gru_model, embeddings, args.seq_len, device)
    is_anomaly = anomaly_prob >= args.threshold

    print(f"\n{'🚨' if is_anomaly else '✅'} Anomaly probability: {anomaly_prob:.4f}")
    print(f"   Verdict: {'ANOMALY DETECTED' if is_anomaly else 'Normal'}")

    # Annotate and save frames
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎨 Annotating frames...")
    annotated_frames = []
    for i, frame in enumerate(frames):
        # Run YOLO detection for bounding boxes
        results = yolo_model(frame, conf=args.conf, verbose=False)
        annotated = annotate_frame(frame, results, anomaly_prob, args.threshold)
        annotated_frames.append(annotated)

        # Save frame
        cv2.imwrite(str(output_dir / f"frame_{i:05d}.jpg"), annotated)

    # Optionally save as video
    if args.save_video and annotated_frames:
        h, w = annotated_frames[0].shape[:2]
        video_path = output_dir / "annotated_output.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
        for frame in annotated_frames:
            writer.write(frame)
        writer.release()
        print(f"  🎬 Video saved: {video_path}")

    print(f"\n  ✅ {len(annotated_frames)} annotated frames saved to: {output_dir}")


if __name__ == "__main__":
    main()
