"""
Stage 2 — Extract feature embeddings from UCF-Crime frames using a
fine-tuned YOLO model.

The script walks datasets/ufc-crime/{Train,Test}/<Category>/<frames>.png,
groups frames by video-clip name, runs model.embed() on each frame,
and saves per-clip feature matrices as .npy files.

Usage:
    uv run python extract_features.py
    uv run python extract_features.py --model runs/detect/train/weights/best.pt --batch-size 32
    uv run python extract_features.py --help
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Categories that count as "Normal" (label=0); all others are "Anomaly" (label=1)
NORMAL_CATEGORIES = {"NormalVideos"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract YOLO embeddings from UCF-Crime frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to fine-tuned YOLO weights",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="datasets/ufc-crime",
        help="Root of UCF-Crime dataset",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="features",
        help="Output directory for .npy files",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction",
    )
    p.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device (0, cpu, etc.)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames per clip (0 = all frames)",
    )
    return p.parse_args()


def extract_clip_name(filename: str) -> str:
    """Extract the video clip identifier from a frame filename.

    Examples:
        'Abuse001_x264_1580.png' -> 'Abuse001_x264'
        'Normal_Videos001_x264_30.png' -> 'Normal_Videos001_x264'
    """
    # Remove extension and split off the trailing frame number
    stem = Path(filename).stem
    # Match everything up to the last underscore followed by digits
    match = re.match(r"(.+)_(\d+)$", stem)
    if match:
        return match.group(1)
    return stem


def group_frames_by_clip(
    category_dir: Path,
) -> dict[str, list[Path]]:
    """Group all frame images in a category folder by their clip name."""
    clips: dict[str, list[Path]] = defaultdict(list)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    for img_path in sorted(category_dir.iterdir()):
        if img_path.suffix.lower() in image_extensions:
            clip_name = extract_clip_name(img_path.name)
            clips[clip_name].append(img_path)

    # Sort frames within each clip by frame number
    for clip_name in clips:
        clips[clip_name].sort(
            key=lambda p: int(re.search(r"_(\d+)$", p.stem).group(1))
            if re.search(r"_(\d+)$", p.stem)
            else 0
        )

    return dict(clips)


def extract_embeddings_batch(
    model: YOLO,
    frame_paths: list[Path],
    batch_size: int,
    max_frames: int,
) -> np.ndarray:
    """Extract embeddings for a list of frames in batches.

    Returns:
        np.ndarray of shape (T, D) where T = number of frames, D = embedding dim.
    """
    if max_frames > 0:
        # Subsample uniformly if we have more frames than max
        if len(frame_paths) > max_frames:
            indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]

    all_embeddings = []

    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i : i + batch_size]
        batch_strs = [str(p) for p in batch_paths]

        # model.embed() returns list of tensors, one per image
        embeddings = model.embed(batch_strs)

        for emb in embeddings:
            # Convert to numpy, flatten if needed
            if hasattr(emb, "cpu"):
                emb_np = emb.cpu().numpy()
            else:
                emb_np = np.array(emb)

            # Flatten to 1D
            emb_np = emb_np.flatten()
            all_embeddings.append(emb_np)

    return np.stack(all_embeddings, axis=0)


def main() -> None:
    args = parse_args()

    # Validate paths
    model_path = Path(args.model)
    data_dir = Path(args.data_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}\n"
            "Train the model first with: uv run python train_yolo.py"
        )
    if not data_dir.exists():
        raise FileNotFoundError(f"UCF-Crime dataset not found: {data_dir}")

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  DroneAI — Stage 2: Feature Extraction (UCF-Crime)")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Data:       {args.data_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    # Load model
    model = YOLO(str(model_path))

    manifest: dict[str, dict] = {}
    total_clips = 0
    total_frames = 0

    for split in ["Train", "Test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"⚠️  Split directory not found: {split_dir}, skipping...")
            continue

        categories = sorted(
            [d for d in split_dir.iterdir() if d.is_dir()]
        )
        print(f"\n📂 Processing {split} split ({len(categories)} categories)...")

        for cat_dir in categories:
            category = cat_dir.name
            is_normal = category in NORMAL_CATEGORIES
            label = 0 if is_normal else 1
            label_str = "Normal" if is_normal else "Anomaly"

            clips = group_frames_by_clip(cat_dir)
            if not clips:
                continue

            print(
                f"  📁 {category} ({label_str}): {len(clips)} clips"
            )

            # Create output directory
            out_cat_dir = output_dir / split / category
            out_cat_dir.mkdir(parents=True, exist_ok=True)

            for clip_name, frame_paths in tqdm(
                clips.items(),
                desc=f"    {category}",
                leave=False,
            ):
                # Extract embeddings
                features = extract_embeddings_batch(
                    model=model,
                    frame_paths=frame_paths,
                    batch_size=args.batch_size,
                    max_frames=args.max_frames,
                )

                # Save .npy
                npy_path = out_cat_dir / f"{clip_name}.npy"
                np.save(str(npy_path), features)

                # Record in manifest
                manifest[f"{split}/{category}/{clip_name}"] = {
                    "npy_path": str(npy_path),
                    "label": label,
                    "label_str": label_str,
                    "category": category,
                    "split": split.lower(),
                    "num_frames": features.shape[0],
                    "embedding_dim": features.shape[1],
                }

                total_clips += 1
                total_frames += features.shape[0]

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  ✅ Feature extraction complete!")
    print(f"  📊 Total clips:  {total_clips}")
    print(f"  🖼️  Total frames: {total_frames}")
    if total_clips > 0:
        sample = next(iter(manifest.values()))
        print(f"  📐 Embedding dim: {sample['embedding_dim']}")
    print(f"  📄 Manifest:     {manifest_path}")
    print("=" * 60)
    print("\n🚀 Next step: train GRU classifier with train_classifier.py")


if __name__ == "__main__":
    main()
