"""
DroneAI — Gradio WebUI for anomaly detection.

Provides a modern interface for:
  1. Single video/image analysis with YOLO + GRU
  2. Batch processing of multiple clips
  3. Pipeline info and usage guide

Usage:
    uv run python app.py
"""

import json
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

from train_classifier import AnomalyGRU

# ---------------------------------------------------------------------------
# Globals (loaded lazily)
# ---------------------------------------------------------------------------
_yolo_model: YOLO | None = None
_gru_model: AnomalyGRU | None = None
_gru_config: dict | None = None
_device: torch.device | None = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _find_models() -> tuple[list[str], list[str]]:
    """Auto-discover available YOLO and GRU model files."""
    yolo_models = sorted(
        [str(p) for p in Path("runs").rglob("*.pt") if "weights" in str(p)]
    ) if Path("runs").exists() else []

    gru_models = sorted(
        [str(p) for p in Path("models").rglob("*.pt")]
    ) if Path("models").exists() else []

    return yolo_models, gru_models


def _load_yolo(model_path: str) -> YOLO:
    global _yolo_model
    if _yolo_model is None or True:  # Always reload on path change
        _yolo_model = YOLO(model_path)
    return _yolo_model


def _load_gru(model_path: str) -> tuple[AnomalyGRU, dict]:
    global _gru_model, _gru_config
    device = _get_device()
    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    _gru_config = {
        "input_dim": ckpt["input_dim"],
        "hidden_size": ckpt["hidden_size"],
        "num_layers": ckpt["num_layers"],
        "dropout": ckpt["dropout"],
        "bidirectional": ckpt["bidirectional"],
        "seq_len": ckpt.get("seq_len", 64),
    }

    _gru_model = AnomalyGRU(
        input_dim=_gru_config["input_dim"],
        hidden_size=_gru_config["hidden_size"],
        num_layers=_gru_config["num_layers"],
        dropout=_gru_config["dropout"],
        bidirectional=_gru_config["bidirectional"],
    ).to(device)
    _gru_model.load_state_dict(ckpt["model_state_dict"])
    _gru_model.eval()

    return _gru_model, _gru_config


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------
def _extract_embeddings(yolo: YOLO, frames: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
    """Extract embeddings from frames using YOLO model."""
    all_embs = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        embeddings = yolo.embed(batch)
        for emb in embeddings:
            arr = emb.cpu().numpy() if hasattr(emb, "cpu") else np.array(emb)
            all_embs.append(arr.flatten())
    return np.stack(all_embs, axis=0)


def _predict(gru: AnomalyGRU, embeddings: np.ndarray, seq_len: int) -> float:
    """Run GRU prediction and return anomaly probability."""
    device = _get_device()
    T, D = embeddings.shape

    if T >= seq_len:
        indices = np.linspace(0, T - 1, seq_len, dtype=int)
        features = embeddings[indices]
    else:
        pad = np.zeros((seq_len - T, D), dtype=embeddings.dtype)
        features = np.concatenate([embeddings, pad], axis=0)

    x = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = gru(x)
        return torch.sigmoid(logits).item()


def _annotate_frame(
    frame: np.ndarray,
    yolo_results,
    anomaly_prob: float,
    threshold: float,
) -> np.ndarray:
    """Annotate a frame with YOLO boxes and anomaly status overlay."""
    out = frame.copy()

    # Draw detections
    if yolo_results and len(yolo_results) > 0:
        r = yolo_results[0]
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = r.names.get(cls, str(cls))
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    out, f"{name} {conf:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )

    # Anomaly banner
    h, w = out.shape[:2]
    is_anomaly = anomaly_prob >= threshold
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0, dst=out)

    color = (0, 0, 255) if is_anomaly else (0, 220, 0)
    icon = "ANOMALY" if is_anomaly else "NORMAL"
    cv2.putText(
        out, f"{icon}  {anomaly_prob:.1%}", (12, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
    )

    # Prob bar
    bar_w = int((w - 30) * anomaly_prob)
    cv2.rectangle(out, (15, 44), (15 + bar_w, 48), color, -1)
    cv2.rectangle(out, (15, 44), (w - 15, 48), (80, 80, 80), 1)

    return out


def _load_video_frames(video_path: str, max_frames: int = 300) -> list[np.ndarray]:
    """Load frames from a video file, capping at max_frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames) if total > max_frames else 1

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------
def analyze_video(
    video_path: str,
    yolo_path: str,
    gru_path: str,
    threshold: float,
    conf: float,
    progress=gr.Progress(),
):
    """Analyze a single video (main tab callback)."""
    if not video_path:
        return None, "[!] Please upload a video.", None

    if not yolo_path or not Path(yolo_path).exists():
        return None, "[!] YOLO model not found. Train Stage 1 first.", None

    if not gru_path or not Path(gru_path).exists():
        return None, "[!] GRU model not found. Train Stage 3 first.", None

    progress(0.1, desc="Loading models...")
    yolo = _load_yolo(yolo_path)
    gru, cfg = _load_gru(gru_path)

    progress(0.2, desc="Loading video frames...")
    frames = _load_video_frames(video_path, max_frames=200)
    if not frames:
        return None, "[!] Could not load any frames from the video.", None

    progress(0.4, desc=f"Extracting embeddings from {len(frames)} frames...")
    embeddings = _extract_embeddings(yolo, frames)

    progress(0.7, desc="Running anomaly classification...")
    prob = _predict(gru, embeddings, cfg["seq_len"])
    is_anomaly = prob >= threshold

    progress(0.8, desc="Annotating sample frames...")
    # Pick up to 8 representative frames for the gallery
    n_gallery = min(8, len(frames))
    gallery_indices = np.linspace(0, len(frames) - 1, n_gallery, dtype=int)
    gallery_images = []

    for gi in gallery_indices:
        frame = frames[gi]
        results = yolo(frame, conf=conf, verbose=False)
        annotated = _annotate_frame(frame, results, prob, threshold)
        # Convert BGR → RGB for Gradio
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        gallery_images.append(annotated_rgb)

    # Save annotated video
    progress(0.9, desc="Saving annotated video...")
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(tmp_video.name, cv2.VideoWriter.fourcc(*"mp4v"), 24, (w, h))

    for i, frame in enumerate(frames):
        results = yolo(frame, conf=conf, verbose=False)
        annotated = _annotate_frame(frame, results, prob, threshold)
        writer.write(annotated)
    writer.release()

    # Build result text
    verdict_emoji = "[ALERT]" if is_anomaly else "[OK]"
    status_text = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
    result_md = f"""
## {verdict_emoji} {status_text}

| Metric | Value |
|--------|-------|
| **Anomaly Probability** | `{prob:.4f}` ({prob:.1%}) |
| **Threshold** | `{threshold}` |
| **Verdict** | **{status_text}** |
| **Frames Analyzed** | {len(frames)} |
| **Embedding Shape** | `{embeddings.shape}` |
| **Device** | `{_get_device()}` |
"""

    progress(1.0, desc="Done!")
    return gallery_images, result_md, tmp_video.name


def analyze_image(
    image: np.ndarray,
    yolo_path: str,
    conf: float,
):
    """Run YOLO detection on a single image (no GRU — just object detection)."""
    if image is None:
        return None, "[!] Please upload an image."

    if not yolo_path or not Path(yolo_path).exists():
        return None, "[!] YOLO model not found."

    yolo = _load_yolo(yolo_path)
    # Gradio gives RGB, YOLO expects BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = yolo(bgr, conf=conf, verbose=False)

    # Annotate
    annotated = bgr.copy()
    det_count = 0
    class_counts: dict[str, int] = {}

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            c = box.conf[0].item()
            cls = int(box.cls[0].item())
            name = results[0].names.get(cls, str(cls))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"{name} {c:.2f}", (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
            det_count += 1
            class_counts[name] = class_counts.get(name, 0) + 1

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(class_counts.items()))
    result_md = f"""
## Detection Results

| Metric | Value |
|--------|-------|
| **Objects Detected** | {det_count} |
| **Classes** | {counts_str or 'None'} |
| **Confidence Threshold** | {conf} |
"""

    return annotated_rgb, result_md


def batch_process(
    folder_path: str,
    yolo_path: str,
    gru_path: str,
    threshold: float,
    progress=gr.Progress(),
):
    """Batch-process a directory of video clips."""
    if not folder_path or not Path(folder_path).exists():
        return "[!] Folder not found."
    if not yolo_path or not Path(yolo_path).exists():
        return "[!] YOLO model not found."
    if not gru_path or not Path(gru_path).exists():
        return "[!] GRU model not found."

    yolo = _load_yolo(yolo_path)
    gru, cfg = _load_gru(gru_path)

    folder = Path(folder_path)
    # Find video files or subdirectories with images
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}
    items = []

    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in video_exts:
            items.append(("video", f))
        elif f.is_dir():
            imgs = [p for p in f.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            if imgs:
                items.append(("frames", f))

    if not items:
        return "[!] No videos or frame folders found in the directory."

    results_md = "## Batch Analysis Results\n\n"
    results_md += "| # | Clip | Type | Frames | Prob | Verdict |\n"
    results_md += "|---|------|------|--------|------|---------|" + "\n"

    for i, (item_type, item_path) in enumerate(items):
        progress((i + 1) / len(items), desc=f"Processing {item_path.name}...")

        if item_type == "video":
            frames = _load_video_frames(str(item_path), max_frames=100)
        else:
            exts = {".png", ".jpg", ".jpeg"}
            frame_paths = sorted(p for p in item_path.iterdir() if p.suffix.lower() in exts)
            frames = [cv2.imread(str(p)) for p in frame_paths[:100]]
            frames = [f for f in frames if f is not None]

        if not frames:
            results_md += f"| {i+1} | {item_path.name} | {item_type} | 0 | -- | Error |\n"
            continue

        embeddings = _extract_embeddings(yolo, frames)
        prob = _predict(gru, embeddings, cfg["seq_len"])
        is_anomaly = prob >= threshold
        emoji = "[!]" if is_anomaly else "[OK]"
        verdict = "ANOMALY" if is_anomaly else "Normal"

        results_md += (
            f"| {i+1} | {item_path.name} | {item_type} | "
            f"{len(frames)} | {prob:.4f} | {emoji} {verdict} |\n"
        )

    anomaly_count = results_md.count("[!]")
    total_count = len(items)
    results_md += f"\n\n**Summary:** {anomaly_count}/{total_count} clips flagged as anomalous.\n"

    return results_md


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* DroneAI Theme */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}

/* Header */
#header-row {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 12px;
    padding: 20px 30px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(48, 43, 99, 0.3);
}

#header-row h1 {
    margin: 0;
    font-size: 1.8em;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

#header-row p {
    color: #a0a0c0;
    margin: 4px 0 0;
    font-size: 0.95em;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1em !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    box-shadow: 0 3px 12px rgba(102, 126, 234, 0.3) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

/* Result cards */
.result-box {
    border: 1px solid #3a3a5c;
    border-radius: 10px;
    padding: 16px;
    background: #1a1a2e;
}

/* Gallery cells */
.gallery-item {
    border-radius: 8px;
    overflow: hidden;
}
"""


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    yolo_models, gru_models = _find_models()
    default_yolo = yolo_models[0] if yolo_models else "runs/detect/train/weights/best.pt"
    default_gru = gru_models[0] if gru_models else "models/gru_best.pt"

    with gr.Blocks(
        title="DroneAI — Anomaly Detection",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
    ) as app:
        # Header
        with gr.Row(elem_id="header-row"):
            gr.Markdown(
                """
# 🚁 DroneAI — Anomaly Detection System
Drone-as-First-Responder • YOLOv26 + GRU Pipeline
                """,
            )

        with gr.Tabs():
            # ═══════════════════════════════════════════════════════════
            # Tab 1: Video Analysis
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("🎬 Video Analysis", id="video-tab"):
                gr.Markdown("Upload a video to detect objects and classify anomalies.")

                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Upload Video",
                            sources=["upload"],
                        )
                        with gr.Accordion("⚙️ Model Settings", open=False):
                            yolo_dropdown = gr.Textbox(
                                label="YOLO Model Path",
                                value=default_yolo,
                                info="Path to fine-tuned YOLO weights",
                            )
                            gru_dropdown = gr.Textbox(
                                label="GRU Model Path",
                                value=default_gru,
                                info="Path to trained GRU checkpoint",
                            )
                        with gr.Accordion("🎛️ Detection Settings", open=False):
                            threshold_slider = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05,
                                label="Anomaly Threshold",
                                info="Probability above this = anomaly",
                            )
                            conf_slider = gr.Slider(
                                0.0, 1.0, value=0.25, step=0.05,
                                label="YOLO Confidence",
                                info="Minimum detection confidence",
                            )

                        analyze_btn = gr.Button(
                            "🔍 Analyze Video",
                            variant="primary",
                            elem_classes="primary-btn",
                        )

                    with gr.Column(scale=2):
                        result_gallery = gr.Gallery(
                            label="Annotated Frames",
                            columns=4,
                            height=400,
                        )
                        result_md = gr.Markdown(
                            value="*Upload a video and click Analyze to see results.*",
                            elem_classes="result-box",
                        )
                        result_video = gr.Video(label="Annotated Video")

                analyze_btn.click(
                    fn=analyze_video,
                    inputs=[video_input, yolo_dropdown, gru_dropdown, threshold_slider, conf_slider],
                    outputs=[result_gallery, result_md, result_video],
                )

            # ═══════════════════════════════════════════════════════════
            # Tab 2: Image Detection
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("🖼️ Image Detection", id="image-tab"):
                gr.Markdown(
                    "Upload a single image to run YOLO object detection "
                    "(no temporal anomaly analysis — GRU requires video sequences)."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(label="Upload Image", type="numpy")
                        img_yolo_path = gr.Textbox(
                            label="YOLO Model Path",
                            value=default_yolo,
                        )
                        img_conf = gr.Slider(
                            0.0, 1.0, value=0.25, step=0.05,
                            label="Confidence Threshold",
                        )
                        detect_btn = gr.Button(
                            "🔎 Detect Objects",
                            variant="primary",
                            elem_classes="primary-btn",
                        )

                    with gr.Column(scale=2):
                        img_output = gr.Image(label="Annotated Image")
                        img_result_md = gr.Markdown(
                            value="*Upload an image and click Detect.*",
                            elem_classes="result-box",
                        )

                detect_btn.click(
                    fn=analyze_image,
                    inputs=[image_input, img_yolo_path, img_conf],
                    outputs=[img_output, img_result_md],
                )

            # ═══════════════════════════════════════════════════════════
            # Tab 3: Batch Processing
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("📊 Batch Process", id="batch-tab"):
                gr.Markdown(
                    "Point to a folder containing video files or subdirectories of frames. "
                    "Each item will be analyzed independently."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_folder = gr.Textbox(
                            label="Folder Path",
                            placeholder="/path/to/clips_folder",
                            info="Directory with .mp4 files or subfolders of frame images",
                        )
                        batch_yolo = gr.Textbox(
                            label="YOLO Model Path", value=default_yolo,
                        )
                        batch_gru = gr.Textbox(
                            label="GRU Model Path", value=default_gru,
                        )
                        batch_threshold = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05,
                            label="Anomaly Threshold",
                        )
                        batch_btn = gr.Button(
                            "📊 Run Batch Analysis",
                            variant="primary",
                            elem_classes="primary-btn",
                        )

                    with gr.Column(scale=2):
                        batch_results = gr.Markdown(
                            value="*Enter a folder path and click Run.*",
                            elem_classes="result-box",
                        )

                batch_btn.click(
                    fn=batch_process,
                    inputs=[batch_folder, batch_yolo, batch_gru, batch_threshold],
                    outputs=[batch_results],
                )

            # ═══════════════════════════════════════════════════════════
            # Tab 4: About
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("ℹ️ About", id="about-tab"):
                gr.Markdown("""
## 🚁 DroneAI — Anomaly Detection Pipeline

### Architecture

```
┌──────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  Video Feed  │ ──▶ │  YOLOv26n (Stage 1)│ ──▶ │  GRU Classifier  │
│  or Frames   │     │  Object Detection  │     │  (Stage 3)       │
└──────────────┘     │  + Embedding       │     │  Temporal Anomaly│
                     │  Extraction        │     │  Classification  │
                     └────────────────────┘     └──────────────────┘
                            │                          │
                     Feature Vectors              Anomaly Prob.
                        (T, D)                    [0.0 → 1.0]
```

### Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| **1** | `train_yolo.py` | Fine-tune YOLOv26n on VisDrone dataset for drone-perspective object detection |
| **2** | `extract_features.py` | Extract penultimate-layer embeddings from UCF-Crime frames using the fine-tuned YOLO model |
| **3** | `train_classifier.py` | Train a bidirectional GRU with attention for binary anomaly classification |
| **—** | `inference.py` | CLI-based end-to-end inference on new videos/frames |
| **—** | `app.py` | This Gradio WebUI |

### Quick Start

```bash
# Stage 1: Fine-tune YOLO on VisDrone
uv run python train_yolo.py --epochs 50 --batch 16

# Stage 2: Extract features from UCF-Crime
uv run python extract_features.py --batch-size 32

# Stage 3: Train GRU classifier
uv run python train_classifier.py --epochs 30 --hidden-size 128

# Launch WebUI
uv run python app.py
```

### Dataset Info

- **VisDrone2019-DET**: Drone-perspective images with 10 object classes (pedestrian, car, bus, etc.)
- **UCF-Crime**: 13 anomaly categories + Normal (pre-extracted as PNG frames)

### Model Info

- **YOLO**: YOLOv26n — nano variant optimized for edge deployment
- **GRU**: Bidirectional GRU with attention pooling, trained with class-weighted BCE loss
                """)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
