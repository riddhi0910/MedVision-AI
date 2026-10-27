"""
scripts/balance_and_split_dataset.py
====================================
High-Performance Offline Data Augmentation & 50-50 Class Balancing Engine.

Reads the clean deduplicated splits from data/custom_dataset,
and balances the minority class (NORMAL) in each split (Train, Val, Test)
so that NORMAL == PNEUMONIA (exact 50-50 class parity).

Streams image writes directly to disk to keep RAM usage under 100 MB.
"""

import json
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "custom_dataset"
OUT_DIR = PROJECT_ROOT / "data" / "balanced_dataset"


def apply_clinical_augmentation(img: Image.Image, rng: random.Random) -> Image.Image:
    """Medically valid transforms (safe for chest radiographs)."""
    out = img.convert("RGB")

    # 1. Bilateral horizontal flip
    if rng.random() > 0.5:
        out = ImageOps.mirror(out)

    # 2. Subtle rotation (-8 to +8 degrees)
    angle = rng.uniform(-8.0, 8.0)
    out = out.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    # 3. Mild contrast tweak (0.92 to 1.08)
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.92, 1.08))

    # 4. Mild brightness tweak (0.92 to 1.08)
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.92, 1.08))

    return out


def main():
    print("\n" + "=" * 65)
    print("  [Step 1] Initializing 50-50 Balancing Pipeline...")
    print("=" * 65)

    if not SOURCE_DIR.exists():
        print(f"Error: {SOURCE_DIR} not found.")
        return

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    summary_stats = {}

    for split in ["train", "val", "test"]:
        print(f"\nProcessing split: [{split.upper()}] ...")
        norm_src = SOURCE_DIR / split / "NORMAL"
        pneu_src = SOURCE_DIR / split / "PNEUMONIA"

        norm_dst = OUT_DIR / split / "NORMAL"
        pneu_dst = OUT_DIR / split / "PNEUMONIA"
        norm_dst.mkdir(parents=True, exist_ok=True)
        pneu_dst.mkdir(parents=True, exist_ok=True)

        # 1. Copy all PNEUMONIA images
        pneu_files = sorted(list(pneu_src.glob("*.jpeg")) + list(pneu_src.glob("*.jpg")))
        for f in pneu_files:
            shutil.copy2(f, pneu_dst / f.name)
        n_pneu = len(pneu_files)
        print(f"  - Copied PNEUMONIA: {n_pneu:,} images")

        # 2. Copy all existing NORMAL images
        norm_files = sorted(list(norm_src.glob("*.jpeg")) + list(norm_src.glob("*.jpg")))
        for f in norm_files:
            shutil.copy2(f, norm_dst / f.name)
        n_norm = len(norm_files)
        print(f"  - Copied NORMAL   : {n_norm:,} original images")

        # 3. Generate augmentations to reach exact parity: n_norm + needed == n_pneu
        needed = n_pneu - n_norm
        print(f"  - Generating {needed:,} clinical augmentations for NORMAL to reach 50-50 balance...")

        for i in range(needed):
            src_f = norm_files[i % n_norm]
            with Image.open(src_f) as base_img:
                aug_img = apply_clinical_augmentation(base_img, rng)
                aug_name = f"normal_{split}_aug_{i+1:05d}.jpeg"
                aug_img.save(norm_dst / aug_name, format="JPEG", quality=95)

            if (i + 1) % 1000 == 0 or (i + 1) == needed:
                print(f"    -> Generated {i+1:,}/{needed:,} augmented images")

        summary_stats[split] = {
            "NORMAL": n_norm + needed,
            "PNEUMONIA": n_pneu,
        }

    # Save stats JSON
    with open(OUT_DIR / "balanced_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "=" * 65)
    print("  [SUCCESS] 50-50 BALANCED DATASET GENERATED!")
    print("=" * 65)
    print(f"{'Split':<12} {'NORMAL':<12} {'PNEUMONIA':<12} {'Total':<10} {'Balance':<10}")
    print("-" * 65)

    grand_total = 0
    for split_name in ["train", "val", "test"]:
        norm_c = summary_stats[split_name]["NORMAL"]
        pneu_c = summary_stats[split_name]["PNEUMONIA"]
        tot = norm_c + pneu_c
        grand_total += tot
        ratio = f"{norm_c/tot*100:.0f}% / {pneu_c/tot*100:.0f}%"
        print(f"{split_name:<12} {norm_c:<12,} {pneu_c:<12,} {tot:<10,} {ratio:<10}")

    print("-" * 65)
    total_norm = sum(summary_stats[s]["NORMAL"] for s in summary_stats)
    total_pneu = sum(summary_stats[s]["PNEUMONIA"] for s in summary_stats)
    print(f"{'Total':<12} {total_norm:<12,} {total_pneu:<12,} {grand_total:<10,}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
