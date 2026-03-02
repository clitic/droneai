"""
Stage 2 -- Extract YOLO embeddings from UCF-Crime frames.
Auto-discovers the trained model from runs/.

Usage:
    uv run python src/extract_embeddings.py
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def find_model() -> Path:
    """Return the path to the trained YOLO model."""
    best = Path("runs/detect/visdrone/weights/best.pt")
    if not best.exists():
        print(f"[ERROR] Model not found: {best}")
        print("        Train first: uv run python src/train_detector.py")
        sys.exit(1)
    print(f"  Model: {best}")
    return best


def clip_name(filename: str) -> str:
    """'Abuse001_x264_1580.png' -> 'Abuse001_x264'"""
    m = re.match(r"(.+)_(\d+)$", Path(filename).stem)
    return m.group(1) if m else Path(filename).stem


def group_frames(category_dir: Path) -> dict[str, list[Path]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    clips: dict[str, list[Path]] = defaultdict(list)
    for p in category_dir.iterdir():
        if p.suffix.lower() in exts:
            clips[clip_name(p.name)].append(p)
    for k in clips:
        clips[k].sort(key=lambda p: int(m.group(1)) if (m := re.search(r"_(\d+)$", p.stem)) else 0)
    return dict(clips)


def embed_clip(model: YOLO, paths: list[Path]) -> np.ndarray:
    embs = []
    for i in range(0, len(paths), 16):
        for e in model.embed([str(p) for p in paths[i:i+16]]):
            arr = e.cpu().numpy() if hasattr(e, "cpu") else np.array(e)
            embs.append(arr.flatten())
    return np.stack(embs)


def main() -> None:
    data_dir = Path("datasets/ufc-crime")
    if not data_dir.exists():
        print(f"[ERROR] Dataset not found: {data_dir}")
        sys.exit(1)

    model_path = find_model()

    print("=" * 60)
    print("  DroneAI -- Stage 2: Embedding Extraction")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Data:  {data_dir}")
    print("=" * 60)

    model = YOLO(str(model_path))
    out_dir = Path("features")
    manifest: dict[str, dict] = {}
    total_clips, total_frames = 0, 0

    for split in ["Train", "Test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"[WARN] {split_dir} not found, skipping...")
            continue

        categories = sorted(d for d in split_dir.iterdir() if d.is_dir())
        print(f"\n[*] {split} split ({len(categories)} categories)")

        # Collect all clips for a single progress bar
        all_clips = []
        for cat in categories:
            is_normal = cat.name == "NormalVideos"
            for name, paths in group_frames(cat).items():
                all_clips.append((cat.name, 0 if is_normal else 1, "Normal" if is_normal else "Anomaly", name, paths))

        if not all_clips:
            continue

        for category, label, label_str, name, paths in tqdm(
            all_clips, desc=f"  {split}", unit="clip", ncols=100, file=sys.stdout, dynamic_ncols=False
        ):
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
