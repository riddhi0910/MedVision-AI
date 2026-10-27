"""
scripts/make_physical_splits.py
===============================
Physically reorganizes data/NORMAL and data/PNEUMONIA into standard disk splits:
  data/train/NORMAL       (3,396 images)
  data/train/PNEUMONIA    (3,396 images)
  data/val/NORMAL         (424 images)
  data/val/PNEUMONIA      (424 images)
  data/test/NORMAL        (425 images)
  data/test/PNEUMONIA     (425 images)
"""

import os
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

NORM_SOURCE = DATA_DIR / "NORMAL"
PNEU_SOURCE = DATA_DIR / "PNEUMONIA"


def split_and_move_class(class_name: str, src_dir: Path, rng: random.Random):
    files = sorted(list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.jpg")))
    rng.shuffle(files)

    total = len(files)
    n_train = int(0.80 * total)
    n_val = int(0.10 * total)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    print(f"\nMoving [{class_name}] ({total:,} files):")
    for split_name, flist in splits.items():
        dst_folder = DATA_DIR / split_name / class_name
        dst_folder.mkdir(parents=True, exist_ok=True)
        for f in flist:
            shutil.move(str(f), str(dst_folder / f.name))
        print(f"  -> {split_name}/{class_name}: {len(flist):,} images")


def main():
    print("\n" + "=" * 65)
    print("  Organizing Data into Physical Disk Splits (Option 2)...")
    print("=" * 65)

    if not NORM_SOURCE.exists() or not PNEU_SOURCE.exists():
        print("Error: data/NORMAL or data/PNEUMONIA not found.")
        return

    rng = random.Random(42)

    # 1. Move NORMAL
    split_and_move_class("NORMAL", NORM_SOURCE, rng)

    # 2. Move PNEUMONIA
    split_and_move_class("PNEUMONIA", PNEU_SOURCE, rng)

    # 3. Clean up the now empty source folders
    try:
        NORM_SOURCE.rmdir()
        PNEU_SOURCE.rmdir()
    except Exception:
        pass

    print("\n" + "=" * 65)
    print("  [SUCCESS] PHYSICAL SPLITS CREATED IN data/ !")
    print("=" * 65)
    for split_name in ["train", "val", "test"]:
        n_c = len(list((DATA_DIR / split_name / "NORMAL").glob("*")))
        p_c = len(list((DATA_DIR / split_name / "PNEUMONIA").glob("*")))
        tot = n_c + p_c
        print(f"  data/{split_name:<6} : NORMAL={n_c:<5,} | PNEUMONIA={p_c:<5,} | Total={tot:<5,} (50% / 50%)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
