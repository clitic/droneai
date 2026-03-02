"""
DroneAI -- Gradio WebUI for anomaly detection.

Usage:
    uv run python src/app.py
"""

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

from train_anomaly import AnomalyGRU

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_yolo: YOLO | None = None
_gru: AnomalyGRU | None = None
_gru_cfg: dict | None = None
_dev: torch.device | None = None


def _device():
    global _dev
    if _dev is None:
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _dev


def _load_yolo(p):
    global _yolo
    _yolo = YOLO(p)
    return _yolo


def _load_gru(p):
    global _gru, _gru_cfg
    d = _device()
    ckpt = torch.load(p, map_location=d, weights_only=True)
    _gru_cfg = {k: ckpt[k] for k in ("input_dim", "hidden_size", "num_layers", "dropout", "bidirectional")}
    _gru_cfg["seq_len"] = ckpt.get("seq_len", 64)
    _gru = AnomalyGRU(
        input_dim=_gru_cfg["input_dim"], hidden_size=_gru_cfg["hidden_size"],
        num_layers=_gru_cfg["num_layers"], dropout=_gru_cfg["dropout"],
        bidirectional=_gru_cfg["bidirectional"],
    ).to(d)
    _gru.load_state_dict(ckpt["model_state_dict"])
    _gru.eval()
    return _gru, _gru_cfg


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def _embed(yolo, frames, bs=16):
    embs = []
    for i in range(0, len(frames), bs):
        for e in yolo.embed(frames[i:i+bs]):
            embs.append((e.cpu().numpy() if hasattr(e, "cpu") else np.array(e)).flatten())
    return np.stack(embs)


def _classify(gru, embs, seq_len):
    d = _device()
    T, D = embs.shape
    feat = embs[np.linspace(0, T-1, seq_len, dtype=int)] if T >= seq_len else np.concatenate([embs, np.zeros((seq_len-T, D), dtype=embs.dtype)])
    with torch.no_grad():
        return torch.sigmoid(gru(torch.from_numpy(feat).float().unsqueeze(0).to(d))).item()


# Per-class color palette (BGR) -- 10 distinct colors for VisDrone classes
_PALETTE = [
    (0, 255, 255),   # yellow
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 165, 255),   # orange
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (0, 128, 255),   # dark orange
    (203, 192, 255), # pink
    (0, 215, 255),   # gold
    (180, 105, 255), # hot pink
]


def _class_color(cls_id: int) -> tuple:
    return _PALETTE[cls_id % len(_PALETTE)]


def _draw(frame, res, prob, thresh):
    out = frame.copy()
    if res and res[0].boxes is not None:
        for b in res[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(b.cls[0].item())
            nm = res[0].names.get(cls_id, "?")
            col = _class_color(cls_id)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, f"{nm} {b.conf[0].item():.2f}", (x1, max(y1-8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    h, w = out.shape[:2]
    bad = prob >= thresh
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (w, 52), (0, 0, 0), -1)
    out = cv2.addWeighted(ov, 0.8, out, 0.2, 0, dst=out)
    color = (0, 0, 255) if bad else (0, 200, 0)
    cv2.putText(out, f"{'ANOMALY' if bad else 'NORMAL'}  {prob:.1%}", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    bw = int((w - 24) * prob)
    cv2.rectangle(out, (12, 44), (12 + bw, 49), color, -1)
    cv2.rectangle(out, (12, 44), (w - 12, 49), (100, 100, 100), 1)
    return out


def _read_video(path, max_frames=300):
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
def analyze_video(video, thresh, conf, progress=gr.Progress()):
    yolo_p = "runs/detect/visdrone/weights/best.pt"
    gru_p = "models/gru_best.pt"

    if not video:
        return None, "Upload a video to begin.", None
    if not Path(yolo_p).exists():
        return None, "YOLO model not found. Run train_detector.py first.", None
    if not Path(gru_p).exists():
        return None, "GRU model not found. Run train_anomaly.py first.", None

    progress(0.1, desc="Loading models")
    yolo = _load_yolo(yolo_p)
    gru, cfg = _load_gru(gru_p)

    progress(0.2, desc="Reading video")
    frames = _read_video(video, 200)
    if not frames:
        return None, "Could not read frames from video.", None

    progress(0.4, desc=f"Embedding {len(frames)} frames")
    embs = _embed(yolo, frames)

    progress(0.7, desc="Classifying")
    prob = _classify(gru, embs, cfg["seq_len"])
    anom = prob >= thresh

    progress(0.8, desc="Annotating")
    idxs = np.linspace(0, len(frames)-1, min(8, len(frames)), dtype=int)
    gallery = [cv2.cvtColor(_draw(frames[i], yolo(frames[i], conf=conf, verbose=False), prob, thresh),
               cv2.COLOR_BGR2RGB) for i in idxs]

    progress(0.9, desc="Encoding video")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(tmp.name, cv2.VideoWriter.fourcc(*"mp4v"), 24, (w, h))
    for f in frames:
        wr.write(_draw(f, yolo(f, conf=conf, verbose=False), prob, thresh))
    wr.release()

    verdict = "ANOMALY DETECTED" if anom else "NORMAL"
    md = f"""### {verdict}

**Probability:** {prob:.4f} ({prob:.1%})  
**Threshold:** {thresh}  
**Frames analyzed:** {len(frames)}  
**Embedding size:** {embs.shape[0]} x {embs.shape[1]}  
**Device:** {_device()}"""

    progress(1.0)
    return gallery, md, tmp.name


def detect_image(image, conf):
    yolo_p = "runs/detect/visdrone/weights/best.pt"

    if image is None:
        return None, "Upload an image to begin."
    if not Path(yolo_p).exists():
        return None, "YOLO model not found. Run train_detector.py first."

    yolo = _load_yolo(yolo_p)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    res = yolo(bgr, conf=conf, verbose=False)

    out, det, counts = bgr.copy(), 0, {}
    if res and res[0].boxes is not None:
        for b in res[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(b.cls[0].item())
            nm = res[0].names.get(cls_id, "?")
            col = _class_color(cls_id)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, f"{nm} {b.conf[0].item():.2f}", (x1, max(y1-8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            det += 1
            counts[nm] = counts.get(nm, 0) + 1

    cs = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    md = f"""### {det} object{'s' if det != 1 else ''} detected

**Classes:** {cs or 'None'}  
**Confidence threshold:** {conf}"""

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), md


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:

    with gr.Blocks(title="DroneAI") as app:

        gr.Markdown("## DroneAI")

        with gr.Tabs():

            # -- Video --
            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=240):
                        vid_in = gr.Video(label="Input", sources=["upload"], height=180)
                        thresh_v = gr.Slider(0, 1, value=0.5, step=0.05, label="Threshold")
                        conf_v = gr.Slider(0, 1, value=0.25, step=0.05, label="Confidence")
                        btn_v = gr.Button("Analyze", variant="primary")
                        result_md = gr.Markdown("_Upload a video._")
                    with gr.Column(scale=3):
                        gallery = gr.Gallery(label="Frames", columns=4, object_fit="cover")
                        result_vid = gr.Video(label="Output")

                btn_v.click(analyze_video, [vid_in, thresh_v, conf_v], [gallery, result_md, result_vid])

            # -- Image --
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=240):
                        img_in = gr.Image(label="Input", type="numpy")
                        conf_i = gr.Slider(0, 1, value=0.25, step=0.05, label="Confidence")
                        btn_i = gr.Button("Detect", variant="primary")
                        img_md = gr.Markdown("_Upload an image._")
                    with gr.Column(scale=3):
                        img_out = gr.Image(label="Result")

                btn_i.click(detect_image, [img_in, conf_i], [img_out, img_md])

    return app


if __name__ == "__main__":
    build_app().launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Monochrome(font=gr.themes.GoogleFont("Inter")),
    )
