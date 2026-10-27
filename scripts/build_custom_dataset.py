"""
scripts/build_custom_dataset.py
===============================
Independent Data Engineering & Curation Engine for MedVision AI.

Takes raw image pool from Kaggle, performs:
1. Integrity check (drops corrupt / truncated images)
2. MD5 Content Hash Deduplication (removes duplicate images to prevent data leakage)
3. Custom 80 / 10 / 10 Stratified Split (80% Train, 10% Val, 10% Test)
4. Saves organized dataset to: data/custom_dataset/
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_SOURCE_DIR = PROJECT_ROOT / "data" / "chest_xray"
CUSTOM_OUT_DIR = PROJECT_ROOT / "data" / "custom_dataset"


def compute_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of image file contents."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_image(filepath: Path) -> bool:
    """Verify image can be opened and verified without errors."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        # Verify it can be converted to RGB
        with Image.open(filepath) as img:
            _ = img.convert("RGB")
        return True
    except Exception:
        return False


def collect_unique_images(source_dir: Path) -> Dict[str, List[Path]]:
    """Scan all images, drop corrupt files, and remove duplicates."""
    print("\n[Step 1] Scanning and validating all raw images in data pool...")
    classes = ["NORMAL", "PNEUMONIA"]
    valid_unique: Dict[str, List[Path]] = {cls: [] for cls in classes}
    seen_hashes: Set[str] = set()

    corrupt_count = 0
    duplicate_count = 0

    exts = ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG")

    for cls in classes:
        print(f"  Scanning class '{cls}'...")
        # Search anywhere in source directory for this class folder
        class_folders = [p for p in source_dir.rglob(cls) if p.is_dir()]
        
        for folder in class_folders:
            for ext in exts:
                for img_path in folder.glob(ext):
                    if not is_valid_image(img_path):
                        corrupt_count += 1
                        continue

                    img_hash = compute_file_hash(img_path)
                    if img_hash in seen_hashes:
                        duplicate_count += 1
                        continue

                    seen_hashes.add(img_hash)
                    valid_unique[cls].append(img_path)

    print(f"\n[Step 1 Complete]")
    print(f"  - Corrupt / Unreadable rejected : {corrupt_count}")
    print(f"  - Duplicate images rejected     : {duplicate_count}")
    print(f"  - Unique Clean NORMAL images   : {len(valid_unique['NORMAL']):,}")
    print(f"  - Unique Clean PNEUMONIA images: {len(valid_unique['PNEUMONIA']):,}")
    print(f"  - Total Unique Clean images     : {sum(len(v) for v in valid_unique.values()):,}")

    return valid_unique


def create_stratified_split(
    clean_images: Dict[str, List[Path]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Dict[str, List[Path]]]:
    """Split dataset into train, val, and test while preserving class proportions."""
    print(f"\n[Step 2] Creating Stratified Split: {int(train_ratio*100)}% Train | {int(val_ratio*100)}% Val | {int(test_ratio*100)}% Test...")
    rng = np.random.default_rng(seed)

    splits: Dict[str, Dict[str, List[Path]]] = {
        "train": {"NORMAL": [], "PNEUMONIA": []},
        "val": {"NORMAL": [], "PNEUMONIA": []},
        "test": {"NORMAL": [], "PNEUMONIA": []},
    }

    for cls, paths in clean_images.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits["train"][cls] = shuffled[:n_train]
        splits["val"][cls] = shuffled[n_train:n_train + n_val]
        splits["test"][cls] = shuffled[n_train + n_val:]

    return splits


def copy_or_link_file(src: Path, dst: Path):
    """Link file if possible, or copy if cross-volume."""
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def save_custom_dataset(splits: Dict[str, Dict[str, List[Path]]], out_dir: Path):
    """Write out the clean split files into out_dir."""
    print(f"\n[Step 3] Exporting curated dataset to: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_stats = {}

    for split_name, class_dict in splits.items():
        summary_stats[split_name] = {}
        for cls, file_list in class_dict.items():
            dest_folder = out_dir / split_name / cls
            dest_folder.mkdir(parents=True, exist_ok=True)

            for idx, src_file in enumerate(file_list):
                # Clean unified filename: normal_train_0001.jpeg
                new_filename = f"{cls.lower()}_{split_name}_{idx+1:05d}{src_file.suffix.lower()}"
                copy_or_link_file(src_file, dest_folder / new_filename)

            summary_stats[split_name][cls] = len(file_list)

    # Save stats JSON
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "=" * 60)
    print("  [SUCCESS] CUSTOM CURATED DATASET READY!")
    print("=" * 60)
    print(f"{'Split':<12} {'NORMAL':<12} {'PNEUMONIA':<12} {'Total':<10} {'Proportion':<10}")
    print("-" * 60)

    grand_total = sum(sum(v.values()) for v in summary_stats.values())
    for split_name in ["train", "val", "test"]:
        norm = summary_stats[split_name].get("NORMAL", 0)
        pneu = summary_stats[split_name].get("PNEUMONIA", 0)
        total = norm + pneu
        pct = (total / grand_total * 100) if grand_total else 0
        print(f"{split_name:<12} {norm:<12,} {pneu:<12,} {total:<10,} {pct:.1f}%")

    print("-" * 60)
    print(f"{'Total':<12} {'':<12} {'':<12} {grand_total:<10,}")
    print("=" * 60)


def main():
    if not RAW_SOURCE_DIR.exists():
        print(f"Error: Raw source directory not found: {RAW_SOURCE_DIR}")
        return

    clean_images = collect_unique_images(RAW_SOURCE_DIR)
    splits = create_stratified_split(clean_images, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10)
    save_custom_dataset(splits, CUSTOM_OUT_DIR)


if __name__ == "__main__":
    main()
