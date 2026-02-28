"""
Gradio Web UI for YOLO26 Drone Detection.
Supports image and video upload with adjustable confidence threshold.

Usage:
    uv run main.py
"""

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent / "runs" / "drone_detect" / "weights" / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}\n"
        "Train a model first with: uv run train_yolo.py"
    )

model = YOLO(str(MODEL_PATH))
CLASS_NAMES = model.names

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = [
    (56, 56, 255),   (151, 157, 255), (31, 112, 255),  (29, 178, 255),
    (49, 210, 207),  (10, 249, 72),   (23, 204, 146),  (134, 219, 61),
    (182, 210, 57),  (243, 223, 48),  (255, 175, 35),  (255, 136, 56),
    (255, 73, 56),   (232, 52, 138),  (197, 44, 219),  (145, 47, 219),
    (86, 64, 221),   (37, 183, 255),  (91, 255, 152),  (255, 214, 102),
]


def get_color(cls_id: int) -> tuple:
    return PALETTE[cls_id % len(PALETTE)]


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_boxes(frame: np.ndarray, results) -> tuple[np.ndarray, dict]:
    """Draw bounding boxes on a frame. Returns (annotated_frame, class_counts)."""
    annotated = frame.copy()
    counts: dict[str, int] = {}
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return annotated, counts

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        color = get_color(cls_id)

        # Count detections per class
        counts[cls_name] = counts.get(cls_name, 0) + 1

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated, counts


def format_summary(counts: dict[str, int]) -> str:
    """Format detection counts into a readable summary."""
    if not counts:
        return "No objects detected."
    total = sum(counts.values())
    lines = [f"**Total detections: {total}**\n"]
    for cls_name, count in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"- **{cls_name}**: {count}")
    return "\n".join(lines)


# ── Image detection ───────────────────────────────────────────────────────────
def detect_image(image: np.ndarray, confidence: float) -> tuple[np.ndarray, str]:
    """Run YOLO detection on an uploaded image."""
    if image is None:
        return None, "Please upload an image."

    results = model.predict(image, conf=confidence, imgsz=640, verbose=False)
    annotated, counts = draw_boxes(image, results)

    # Convert BGR → RGB for Gradio display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) if len(annotated.shape) == 3 else annotated

    return annotated_rgb, format_summary(counts)


# ── Video detection ───────────────────────────────────────────────────────────
def detect_video(video_path: str, confidence: float, progress=gr.Progress()) -> tuple[str | None, str]:
    """Run YOLO detection on every frame of an uploaded video."""
    if video_path is None:
        return None, "Please upload a video."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Could not open video file."

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write to a temp file
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out.name, fourcc, fps, (w, h))

    all_counts: dict[str, int] = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=confidence, imgsz=640, verbose=False)
        annotated, counts = draw_boxes(frame, results)
        writer.write(annotated)

        # Accumulate counts
        for cls_name, count in counts.items():
            all_counts[cls_name] = all_counts.get(cls_name, 0) + count

        frame_idx += 1
        progress(frame_idx / total_frames, desc=f"Processing frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()

    summary = format_summary(all_counts)
    summary += f"\n\n*{total_frames} frames • {w}×{h} @ {fps:.0f} fps*"

    return tmp_out.name, summary


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    class_list = ", ".join(CLASS_NAMES.values())

    with gr.Blocks(
        title="DroneAI – YOLO26 Detection",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
        ),
    ) as app:
        gr.Markdown(
            "# 🛩️ DroneAI – YOLO26 Object Detection\n"
            f"**Model:** `{MODEL_PATH.name}` &nbsp;|&nbsp; "
            f"**Classes ({len(CLASS_NAMES)}):** {class_list}"
        )

        with gr.Tab("🖼️ Image Detection"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="Upload Image", type="numpy")
                    img_conf = gr.Slider(
                        minimum=0.05, maximum=1.0, value=0.25, step=0.05,
                        label="Confidence Threshold",
                    )
                    img_btn = gr.Button("🔍 Detect", variant="primary", size="lg")
                with gr.Column():
                    img_output = gr.Image(label="Detection Result")
                    img_summary = gr.Markdown(label="Summary")

            img_btn.click(
                fn=detect_image,
                inputs=[img_input, img_conf],
                outputs=[img_output, img_summary],
            )

        with gr.Tab("🎬 Video Detection"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video")
                    vid_conf = gr.Slider(
                        minimum=0.05, maximum=1.0, value=0.25, step=0.05,
                        label="Confidence Threshold",
                    )
                    vid_btn = gr.Button("🔍 Detect", variant="primary", size="lg")
                with gr.Column():
                    vid_output = gr.Video(label="Detection Result")
                    vid_summary = gr.Markdown(label="Summary")

            vid_btn.click(
                fn=detect_video,
                inputs=[vid_input, vid_conf],
                outputs=[vid_output, vid_summary],
            )

        gr.Markdown(
            "---\n"
            "*Built with [Ultralytics YOLO26](https://docs.ultralytics.com) & "
            "[Gradio](https://gradio.app)*"
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, inbrowser=True)
