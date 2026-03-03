import cv2
import gradio as gr
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

from train_anomaly import AnomalyGRU

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
    _gru_cfg["num_classes"] = ckpt.get("num_classes", 1)
    _gru_cfg["class_names"] = ckpt.get("class_names", ["Normal", "Anomaly"])
    _gru = AnomalyGRU(
        input_dim=_gru_cfg["input_dim"], hidden_size=_gru_cfg["hidden_size"],
        num_layers=_gru_cfg["num_layers"], dropout=_gru_cfg["dropout"],
        bidirectional=_gru_cfg["bidirectional"],
        num_classes=_gru_cfg["num_classes"],
    ).to(d)
    _gru.load_state_dict(ckpt["model_state_dict"])
    _gru.eval()
    return _gru, _gru_cfg


def _embed(yolo, frames, bs=16):
    embs = []
    for i in range(0, len(frames), bs):
        for e in yolo.embed(frames[i:i+bs]):
            embs.append((e.cpu().numpy() if hasattr(e, "cpu") else np.array(e)).flatten())
    return np.stack(embs)


def _classify(gru, embs, cfg):
    d = _device()
    seq_len = cfg["seq_len"]
    class_names = cfg["class_names"]
    T, D = embs.shape
    feat = embs[np.linspace(0, T-1, seq_len, dtype=int)] if T >= seq_len else np.concatenate([embs, np.zeros((seq_len-T, D), dtype=embs.dtype)])
    with torch.no_grad():
        logits = gru(torch.from_numpy(feat).float().unsqueeze(0).to(d))
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    pred_idx = int(probs.argmax())
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
    all_probs = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return pred_name, float(probs[pred_idx]), all_probs


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


def analyze_video(video, conf, progress=gr.Progress()):
    yolo_p = "runs/detect/visdrone/weights/best.pt"
    gru_p = "runs/gru_best.pt"

    if not video:
        return None
    if not Path(yolo_p).exists():
        return None
    if not Path(gru_p).exists():
        return None

    progress(0.1, desc="Loading models")
    yolo = _load_yolo(yolo_p)
    gru, cfg = _load_gru(gru_p)

    progress(0.2, desc="Reading video")
    frames = _read_video(video, 200)
    if not frames:
        return None

    progress(0.4, desc=f"Embedding {len(frames)} frames")
    embs = _embed(yolo, frames)

    progress(0.7, desc="Classifying")
    pred_name, pred_conf, all_probs = _classify(gru, embs, cfg)

    top5 = sorted(all_probs.items(), key=lambda x: -x[1])[:5]
    bar_chart = {name: float(prob) for name, prob in top5}

    progress(1.0)
    return bar_chart


def detect_image(image, conf):
    yolo_p = "runs/detect/visdrone/weights/best.pt"

    if image is None:
        return None
    if not Path(yolo_p).exists():
        return None

    yolo = _load_yolo(yolo_p)
    res = yolo(image, conf=conf, verbose=False)

    if res and hasattr(res[0], "plot"):
        return res[0].plot()
    return image


def build_app() -> gr.Blocks:
    with gr.Blocks(title="DroneAI") as app:
        gr.Markdown("## DroneAI")

        with gr.Tabs():
            with gr.Tab("Video"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        vid_in = gr.Video(label="Input", sources=["upload"])
                        conf_v = gr.Slider(0, 1, value=0.25, step=0.05, label="Detection Confidence")
                        btn_v = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=1):
                        bar_chart = gr.Label(label="Class Probabilities", num_top_classes=5)

                btn_v.click(analyze_video, [vid_in, conf_v], [bar_chart])

            with gr.Tab("Image"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        img_in = gr.Image(label="Input", type="numpy")
                        conf_i = gr.Slider(0, 1, value=0.25, step=0.05, label="Detection Confidence")
                        btn_i = gr.Button("Detect", variant="primary")
                    with gr.Column(scale=1):
                        img_out = gr.Image(label="Result")

                btn_i.click(detect_image, [img_in, conf_i], [img_out])

    return app


if __name__ == "__main__":
    build_app().launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Monochrome(font=gr.themes.GoogleFont("Inter")),
    )
