"""
Stage 2 -- Extract YOLO embeddings from UCF-Crime frames.

Usage:
    uv run python src/extract_embeddings.py
"""

import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def clip_name(filename: str) -> str:
    """'Abuse001_x264_1580.png' -> 'Abuse001_x264'"""
    m = re.match(r"(.+)_(\d+)$", Path(filename).stem)
    return m.group(1) if m else Path(filename).stem


def group_frames(category_dir: Path) -> dict[str, list[Path]]:
    """Group frames by clip name, sorted by frame number."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    clips: dict[str, list[Path]] = defaultdict(list)
    for p in category_dir.iterdir():
        if p.suffix.lower() in exts:
            clips[clip_name(p.name)].append(p)
    for k in clips:
        clips[k].sort(key=lambda p: int(m.group(1)) if (m := re.search(r"_(\d+)$", p.stem)) else 0)
    return dict(clips)


def preload_batch(paths: list[Path]) -> list[np.ndarray]:
    """Load images from disk using threads for faster I/O."""
    def read(p):
        img = cv2.imread(str(p))
        return img if img is not None else np.zeros((64, 64, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(read, paths))


def embed_clip(model: YOLO, paths: list[Path], batch_size: int = 16) -> np.ndarray:
    """Extract embeddings with threaded pre-loading and large batches."""
    embs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        # Pre-load images from disk in parallel
        images = preload_batch(batch_paths)
        # Run embedding on GPU (verbose=False to not pollute tqdm)
        for e in model.embed(images, verbose=False):
            arr = e.cpu().numpy() if hasattr(e, "cpu") else np.array(e)
            embs.append(arr.flatten())
    return np.stack(embs)


def main() -> None:
    # embed() requires PyTorch model — ONNX does not support intermediate layer extraction
    model_path = Path("runs/detect/visdrone/weights/best.pt")
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("        Train first: uv run python src/train_detector.py")
        sys.exit(1)

    data_dir = Path("datasets/ufc-crime")
    if not data_dir.exists():
        print(f"[ERROR] Dataset not found: {data_dir}")
        sys.exit(1)

    # Suppress YOLO logging noise
    os.environ["YOLO_VERBOSE"] = "False"

    print("=" * 60)
    print("  DroneAI -- Stage 2: Embedding Extraction")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Data:  {data_dir}")
    print("=" * 60, flush=True)

    model = YOLO(str(model_path))
    out_dir = Path("datasets/ucf-crime-features")
    manifest: dict[str, dict] = {}
    total_clips, total_frames = 0, 0

    for split in ["Train", "Test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"[WARN] {split_dir} not found, skipping...", flush=True)
            continue

        categories = sorted(d for d in split_dir.iterdir() if d.is_dir())

        # Collect all clips
        all_clips = []
        for cat in categories:
            is_normal = cat.name == "NormalVideos"
            for name, paths in group_frames(cat).items():
                all_clips.append((cat.name, 0 if is_normal else 1,
                                  "Normal" if is_normal else "Anomaly", name, paths))

        if not all_clips:
            continue

        total_clip_frames = sum(len(c[4]) for c in all_clips)
        print(f"\n  {split}: {len(all_clips)} clips, {total_clip_frames} frames", flush=True)

        # Process with progress bar
        done_frames = 0
        pbar = tqdm(total=total_clip_frames, desc=f"  {split}", unit="fr",
                    bar_format="{l_bar}{bar:30}{r_bar}", leave=True)

        for category, label, label_str, name, paths in all_clips:
            (out_dir / split / category).mkdir(parents=True, exist_ok=True)
            features = embed_clip(model, paths)
            npy = out_dir / split / category / f"{name}.npy"
            np.save(str(npy), features)

            manifest[f"{split}/{category}/{name}"] = {
                "npy_path": str(npy), "label": label, "label_str": label_str,
                "category": category, "split": split.lower(),
                "num_frames": features.shape[0], "embedding_dim": features.shape[1],
            }
            total_clips += 1
            total_frames += features.shape[0]
            pbar.update(len(paths))

        pbar.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  [OK] Done! {total_clips} clips, {total_frames} frames")
    if total_clips:
        print(f"  Embedding dim: {next(iter(manifest.values()))['embedding_dim']}")
    print(f"  Manifest: {out_dir / 'manifest.json'}")
    print(f"{'='*60}")
    print("\n>> Next: uv run python src/train_anomaly.py")


if __name__ == "__main__":
    main()
