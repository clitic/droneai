"""
DroneAI -- Gradio WebUI for anomaly detection.

Usage:
    uv run python src/app.py
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

from train_anomaly import AnomalyGRU

# ---------------------------------------------------------------------------
# Globals
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
    yolo = sorted(str(p) for p in Path("runs").rglob("*.pt") if "weights" in str(p)) if Path("runs").exists() else []
    gru = sorted(str(p) for p in Path("models").rglob("*.pt")) if Path("models").exists() else []
    return yolo, gru


def _load_yolo(path: str) -> YOLO:
    global _yolo_model
    _yolo_model = YOLO(path)
    return _yolo_model


def _load_gru(path: str) -> tuple[AnomalyGRU, dict]:
    global _gru_model, _gru_config
    device = _get_device()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    _gru_config = {k: ckpt[k] for k in ("input_dim", "hidden_size", "num_layers", "dropout", "bidirectional")}
    _gru_config["seq_len"] = ckpt.get("seq_len", 64)

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
# Processing
# ---------------------------------------------------------------------------
def _extract_embeddings(yolo: YOLO, frames: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
    embs = []
    for i in range(0, len(frames), batch_size):
        for emb in yolo.embed(frames[i : i + batch_size]):
            arr = emb.cpu().numpy() if hasattr(emb, "cpu") else np.array(emb)
            embs.append(arr.flatten())
    return np.stack(embs, axis=0)


def _predict(gru: AnomalyGRU, embeddings: np.ndarray, seq_len: int) -> float:
    device = _get_device()
    T, D = embeddings.shape
    if T >= seq_len:
        features = embeddings[np.linspace(0, T - 1, seq_len, dtype=int)]
    else:
        features = np.concatenate([embeddings, np.zeros((seq_len - T, D), dtype=embeddings.dtype)])

    x = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(gru(x)).item()


def _annotate(frame: np.ndarray, yolo_results, prob: float, threshold: float) -> np.ndarray:
    out = frame.copy()
    if yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes is not None:
        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            c = box.conf[0].item()
            name = yolo_results[0].names.get(int(box.cls[0].item()), "?")
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{name} {c:.2f}", (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    h, w = out.shape[:2]
    is_bad = prob >= threshold
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0, dst=out)
    color = (0, 0, 255) if is_bad else (0, 220, 0)
    cv2.putText(out, f"{'ANOMALY' if is_bad else 'NORMAL'}  {prob:.1%}", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    bw = int((w - 30) * prob)
    cv2.rectangle(out, (15, 44), (15 + bw, 48), color, -1)
    cv2.rectangle(out, (15, 44), (w - 15, 48), (80, 80, 80), 1)
    return out


def _load_video(path: str, max_frames: int = 300) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames, total = [], int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames) if total > max_frames else 1
    idx = 0
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(f)
        idx += 1
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def analyze_video(video_path, yolo_path, gru_path, threshold, conf, progress=gr.Progress()):
    if not video_path:
        return None, "[!] Please upload a video.", None
    if not yolo_path or not Path(yolo_path).exists():
        return None, "[!] YOLO model not found. Train Stage 1 first.", None
    if not gru_path or not Path(gru_path).exists():
        return None, "[!] GRU model not found. Train Stage 3 first.", None

    progress(0.1, desc="Loading models...")
    yolo = _load_yolo(yolo_path)
    gru, cfg = _load_gru(gru_path)

    progress(0.2, desc="Reading frames...")
    frames = _load_video(video_path, max_frames=200)
    if not frames:
        return None, "[!] Could not read any frames.", None

    progress(0.4, desc=f"Extracting embeddings ({len(frames)} frames)...")
    embeddings = _extract_embeddings(yolo, frames)

    progress(0.7, desc="Classifying...")
    prob = _predict(gru, embeddings, cfg["seq_len"])
    is_anomaly = prob >= threshold

    progress(0.8, desc="Annotating...")
    n_gallery = min(8, len(frames))
    indices = np.linspace(0, len(frames) - 1, n_gallery, dtype=int)
    gallery = []
    for gi in indices:
        res = yolo(frames[gi], conf=conf, verbose=False)
        ann = _annotate(frames[gi], res, prob, threshold)
        gallery.append(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))

    progress(0.9, desc="Saving video...")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter.fourcc(*"mp4v"), 24, (w, h))
    for f in frames:
        writer.write(_annotate(f, yolo(f, conf=conf, verbose=False), prob, threshold))
    writer.release()

    status = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
    tag = "[ALERT]" if is_anomaly else "[OK]"
    md = f"""
## {tag} {status}

| Metric | Value |
|--------|-------|
| **Anomaly Probability** | `{prob:.4f}` ({prob:.1%}) |
| **Threshold** | `{threshold}` |
| **Verdict** | **{status}** |
| **Frames Analyzed** | {len(frames)} |
| **Embedding Shape** | `{embeddings.shape}` |
| **Device** | `{_get_device()}` |
"""
    progress(1.0, desc="Done!")
    return gallery, md, tmp.name


def analyze_image(image, yolo_path, conf):
    if image is None:
        return None, "[!] Upload an image."
    if not yolo_path or not Path(yolo_path).exists():
        return None, "[!] YOLO model not found."

    yolo = _load_yolo(yolo_path)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = yolo(bgr, conf=conf, verbose=False)

    out = bgr.copy()
    det, counts = 0, {}
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            name = results[0].names.get(int(box.cls[0].item()), "?")
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{name} {box.conf[0].item():.2f}", (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            det += 1
            counts[name] = counts.get(name, 0) + 1

    cs = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    md = f"""
## Detection Results

| Metric | Value |
|--------|-------|
| **Objects** | {det} |
| **Classes** | {cs or 'None'} |
| **Confidence** | {conf} |
"""
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), md


def batch_process(folder_path, yolo_path, gru_path, threshold, progress=gr.Progress()):
    if not folder_path or not Path(folder_path).exists():
        return "[!] Folder not found."
    if not yolo_path or not Path(yolo_path).exists():
        return "[!] YOLO model not found."
    if not gru_path or not Path(gru_path).exists():
        return "[!] GRU model not found."

    yolo = _load_yolo(yolo_path)
    gru, cfg = _load_gru(gru_path)

    folder = Path(folder_path)
    vid_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}
    items = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in vid_exts:
            items.append(("video", f))
        elif f.is_dir() and any(p.suffix.lower() in {".png", ".jpg"} for p in f.iterdir()):
            items.append(("frames", f))

    if not items:
        return "[!] No videos or frame folders found."

    md = "## Batch Results\n\n| # | Clip | Type | Frames | Prob | Verdict |\n|---|------|------|--------|------|---------|" + "\n"

    for i, (typ, p) in enumerate(items):
        progress((i + 1) / len(items), desc=f"Processing {p.name}...")
        if typ == "video":
            frames = _load_video(str(p), max_frames=100)
        else:
            frames = [cv2.imread(str(x)) for x in sorted(p.iterdir()) if x.suffix.lower() in {".png", ".jpg"}][:100]
            frames = [f for f in frames if f is not None]

        if not frames:
            md += f"| {i+1} | {p.name} | {typ} | 0 | -- | Error |\n"
            continue

        prob = _predict(gru, _extract_embeddings(yolo, frames), cfg["seq_len"])
        v = "[!] ANOMALY" if prob >= threshold else "[OK] Normal"
        md += f"| {i+1} | {p.name} | {typ} | {len(frames)} | {prob:.4f} | {v} |\n"

    ac = md.count("[!]")
    md += f"\n\n**Summary:** {ac}/{len(items)} clips flagged.\n"
    return md


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
#header-row {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 12px; padding: 20px 30px; margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(48,43,99,0.3);
}
#header-row h1 {
    margin: 0; font-size: 1.8em; font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
#header-row p { color: #a0a0c0; margin: 4px 0 0; font-size: 0.95em; }
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important; font-weight: 600 !important; border-radius: 8px !important;
    box-shadow: 0 3px 12px rgba(102,126,234,0.3) !important;
}
.primary-btn:hover { transform: translateY(-1px) !important; }
.result-box { border: 1px solid #3a3a5c; border-radius: 10px; padding: 16px; background: #1a1a2e; }
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    yolo_models, gru_models = _find_models()
    dy = yolo_models[0] if yolo_models else "runs/detect/visdrone/weights/best.pt"
    dg = gru_models[0] if gru_models else "models/gru_best.pt"

    with gr.Blocks(
        title="DroneAI",
    ) as app:
        with gr.Row(elem_id="header-row"):
            gr.Markdown("# DroneAI -- Anomaly Detection System\nDrone-as-First-Responder | YOLOv26 + GRU Pipeline")

        with gr.Tabs():
            # -- Video --
            with gr.Tab("Video Analysis", id="video-tab"):
                gr.Markdown("Upload a video to detect objects and classify anomalies.")
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_in = gr.Video(label="Upload Video", sources=["upload"])
                        with gr.Accordion("Model Settings", open=False):
                            v_yolo = gr.Textbox(label="YOLO Model", value=dy)
                            v_gru = gr.Textbox(label="GRU Model", value=dg)
                        with gr.Accordion("Detection Settings", open=False):
                            v_thresh = gr.Slider(0, 1, value=0.5, step=0.05, label="Anomaly Threshold")
                            v_conf = gr.Slider(0, 1, value=0.25, step=0.05, label="YOLO Confidence")
                        btn_vid = gr.Button("Analyze Video", variant="primary", elem_classes="primary-btn")
                    with gr.Column(scale=2):
                        gal = gr.Gallery(label="Annotated Frames", columns=4, height=400)
                        res_md = gr.Markdown("*Upload a video and click Analyze.*", elem_classes="result-box")
                        res_vid = gr.Video(label="Annotated Video")
                btn_vid.click(analyze_video, [vid_in, v_yolo, v_gru, v_thresh, v_conf], [gal, res_md, res_vid])

            # -- Image --
            with gr.Tab("Image Detection", id="image-tab"):
                gr.Markdown("Upload a single image for YOLO object detection.")
                with gr.Row():
                    with gr.Column(scale=1):
                        img_in = gr.Image(label="Upload Image", type="numpy")
                        i_yolo = gr.Textbox(label="YOLO Model", value=dy)
                        i_conf = gr.Slider(0, 1, value=0.25, step=0.05, label="Confidence")
                        btn_img = gr.Button("Detect Objects", variant="primary", elem_classes="primary-btn")
                    with gr.Column(scale=2):
                        img_out = gr.Image(label="Result")
                        img_md = gr.Markdown("*Upload an image and click Detect.*", elem_classes="result-box")
                btn_img.click(analyze_image, [img_in, i_yolo, i_conf], [img_out, img_md])

            # -- Batch --
            with gr.Tab("Batch Process", id="batch-tab"):
                gr.Markdown("Analyze a folder of videos or frame directories.")
                with gr.Row():
                    with gr.Column(scale=1):
                        b_folder = gr.Textbox(label="Folder Path", placeholder="/path/to/clips")
                        b_yolo = gr.Textbox(label="YOLO Model", value=dy)
                        b_gru = gr.Textbox(label="GRU Model", value=dg)
                        b_thresh = gr.Slider(0, 1, value=0.5, step=0.05, label="Threshold")
                        btn_batch = gr.Button("Run Batch", variant="primary", elem_classes="primary-btn")
                    with gr.Column(scale=2):
                        b_md = gr.Markdown("*Enter folder and click Run.*", elem_classes="result-box")
                btn_batch.click(batch_process, [b_folder, b_yolo, b_gru, b_thresh], [b_md])

            # -- About --
            with gr.Tab("About", id="about-tab"):
                gr.Markdown("""
## DroneAI -- Anomaly Detection Pipeline

### Architecture
```
Video/Frames --> YOLOv26n (Detection + Embedding) --> GRU (Anomaly Classification)
                    Feature Vectors (T, D)              Probability [0..1]
```

### Pipeline
| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `src/train_detector.py` | Fine-tune YOLOv26n on VisDrone |
| 2 | `src/extract_embeddings.py` | Extract embeddings from UCF-Crime |
| 3 | `src/train_anomaly.py` | Train GRU anomaly classifier |
| UI | `src/app.py` | This Gradio WebUI |

### Quick Start
```bash
uv run python src/train_detector.py
uv run python src/extract_embeddings.py
uv run python src/train_anomaly.py
uv run python src/app.py
```
                """)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate", font=gr.themes.GoogleFont("Inter")),
        css=CSS,
    )
