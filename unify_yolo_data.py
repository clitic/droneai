"""
Unify multiple YOLO26-format datasets from ./data into a single dataset at ./unified_dataset.
Handles whitespace in paths, uses multithreading, and tracks progress with tqdm.
"""

import os
import shutil
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).resolve().parent / "data"
OUTPUT_ROOT = Path(__file__).resolve().parent / "data/unified"
MAX_WORKERS = 16  # number of threads for copying

# ── Discover datasets & build unified class list ──────────────────────────────
def load_dataset_meta(dataset_dir: Path):
    """Return (class_names, splits_present) from a dataset's data.yaml."""
    yaml_path = dataset_dir / "data.yaml"
    with open(str(yaml_path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    # Normalise near-duplicate names (e.g. "Walls" -> "Wall")
    normalised = []
    for n in names:
        n_stripped = n.rstrip("s") if n.endswith("s") and n not in ("Grass", "Stairs", "Windows") else n
        # Special case: "Walls" -> "Wall"
        if n == "Walls":
            n_stripped = "Wall"
        else:
            n_stripped = n
        normalised.append(n_stripped)
    splits = []
    for split_name in ("train", "valid", "test"):
        split_path = dataset_dir / split_name
        if split_path.exists():
            splits.append(split_name)
    return normalised, splits


def build_unified_class_list(datasets_meta):
    """Merge all class names into one ordered list (deterministic)."""
    seen = set()
    unified = []
    for _, (names, _) in sorted(datasets_meta.items()):
        for n in names:
            if n not in seen:
                seen.add(n)
                unified.append(n)
    return unified


# ── Collect copy jobs ─────────────────────────────────────────────────────────
def collect_jobs(datasets_meta, unified_names):
    """
    Walk every dataset/split and build a list of jobs:
        (src_img, dst_img, src_lbl, dst_lbl, class_remap | None)
    class_remap is a dict {old_id: new_id} when the label file needs rewriting.
    """
    jobs = []
    unified_index = {n: i for i, n in enumerate(unified_names)}

    for ds_name, (local_names, splits) in datasets_meta.items():
        ds_dir = DATA_ROOT / ds_name

        # Build remap table: old_class_id -> new_class_id
        remap = {}
        needs_remap = False
        for old_id, local_name in enumerate(local_names):
            new_id = unified_index[local_name]
            remap[old_id] = new_id
            if old_id != new_id:
                needs_remap = True

        for split in splits:
            img_dir = ds_dir / split / "images"
            lbl_dir = ds_dir / split / "labels"
            if not img_dir.exists():
                continue

            out_img_dir = OUTPUT_ROOT / split / "images"
            out_lbl_dir = OUTPUT_ROOT / split / "labels"

            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    continue

                # Use dataset prefix to avoid filename collisions
                safe_prefix = ds_name.replace(" ", "_")
                new_name = f"{safe_prefix}__{img_file.name}"

                dst_img = out_img_dir / new_name

                # Corresponding label
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                dst_lbl = out_lbl_dir / (img_file.stem + ".txt").replace(
                    img_file.stem, f"{safe_prefix}__{img_file.stem}"
                )

                jobs.append((
                    str(img_file),          # src image  (str to handle whitespace)
                    str(dst_img),           # dst image
                    str(lbl_file) if lbl_file.exists() else None,  # src label
                    str(dst_lbl),           # dst label
                    remap if needs_remap else None,
                ))

    return jobs


# ── Worker function ───────────────────────────────────────────────────────────
def process_one(job):
    """Copy an image and optionally remap + copy its label file."""
    src_img, dst_img, src_lbl, dst_lbl, remap = job

    # Ensure parent dirs exist (thread-safe with exist_ok)
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)

    # Copy image
    shutil.copy2(src_img, dst_img)

    # Copy / remap label
    if src_lbl and os.path.isfile(src_lbl):
        if remap is None:
            shutil.copy2(src_lbl, dst_lbl)
        else:
            with open(src_lbl, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_id = int(parts[0])
                parts[0] = str(remap.get(old_id, old_id))
                new_lines.append(" ".join(parts) + "\n")
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
    else:
        # Write an empty label file so YOLO knows there are no objects
        with open(dst_lbl, "w", encoding="utf-8") as f:
            pass

    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("🔍  Scanning datasets …")
    datasets_meta = {}
    for entry in sorted(DATA_ROOT.iterdir()):
        if entry.is_dir() and (entry / "data.yaml").exists():
            names, splits = load_dataset_meta(entry)
            datasets_meta[entry.name] = (names, splits)
            print(f"   • {entry.name}  →  {len(names)} classes, splits: {splits}")

    unified_names = build_unified_class_list(datasets_meta)
    print(f"\n📋  Unified class list ({len(unified_names)} classes):")
    for i, n in enumerate(unified_names):
        print(f"   {i}: {n}")

    print("\n📂  Collecting files …")
    jobs = collect_jobs(datasets_meta, unified_names)
    print(f"   Total files to process: {len(jobs)}")

    # Clean / create output directory
    if OUTPUT_ROOT.exists():
        shutil.rmtree(str(OUTPUT_ROOT))
    OUTPUT_ROOT.mkdir(parents=True)

    print(f"\n🚀  Copying & remapping with {MAX_WORKERS} threads …")
    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_one, job): job for job in jobs}
        with tqdm(total=len(futures), unit="file", desc="Processing") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append((futures[future], exc))
                pbar.update(1)

    # Write unified data.yaml
    splits_present = {}
    for split_name in ("train", "valid", "test"):
        sp = OUTPUT_ROOT / split_name / "images"
        if sp.exists() and any(sp.iterdir()):
            splits_present[split_name] = f"./{split_name}/images"

    unified_yaml = {
        "path": ".",
        "nc": len(unified_names),
        "names": unified_names,
    }
    if "train" in splits_present:
        unified_yaml["train"] = splits_present["train"]
    if "valid" in splits_present:
        unified_yaml["val"] = splits_present["valid"]
    if "test" in splits_present:
        unified_yaml["test"] = splits_present["test"]

    yaml_path = OUTPUT_ROOT / "data.yaml"
    with open(str(yaml_path), "w", encoding="utf-8") as f:
        yaml.dump(unified_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅  Done!  Unified dataset written to: {OUTPUT_ROOT}")
    print(f"   data.yaml → {yaml_path}")
    if errors:
        print(f"\n⚠️  {len(errors)} errors occurred:")
        for job, exc in errors[:10]:
            print(f"   {job[0]} → {exc}")
    else:
        print("   No errors.")


if __name__ == "__main__":
    main()
