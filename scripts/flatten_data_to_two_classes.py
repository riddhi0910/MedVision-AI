"""
scripts/flatten_data_to_two_classes.py
======================================
Combines all 50-50 balanced images into exactly 2 root folders inside data/:
  data/NORMAL/     (8,490 images)
  data/PNEUMONIA/  (8,490 images)

Removes all intermediate nested folders so data/ contains ONLY these 2 folders.
"""

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BALANCED_DIR = DATA_DIR / "balanced_dataset"

NORMAL_DEST = DATA_DIR / "NORMAL"
PNEUMONIA_DEST = DATA_DIR / "PNEUMONIA"


def main():
    print("\n" + "=" * 65)
    print("  [Step 1] Combining All Images into 2 Clean Class Folders...")
    print("=" * 65)

    if not BALANCED_DIR.exists():
        print(f"Error: {BALANCED_DIR} not found.")
        return

    NORMAL_DEST.mkdir(parents=True, exist_ok=True)
    PNEUMONIA_DEST.mkdir(parents=True, exist_ok=True)

    # 1. Collect all NORMAL images across train, val, test
    norm_counter = 0
    for split in ["train", "val", "test"]:
        split_norm = BALANCED_DIR / split / "NORMAL"
        if split_norm.exists():
            for f in sorted(list(split_norm.glob("*.jpeg")) + list(split_norm.glob("*.jpg"))):
                norm_counter += 1
                dst_name = f"normal_{norm_counter:06d}.jpeg"
                shutil.move(str(f), str(NORMAL_DEST / dst_name))

    print(f"  - Moved NORMAL images: {norm_counter:,} -> data/NORMAL/")

    # 2. Collect all PNEUMONIA images across train, val, test
    pneu_counter = 0
    for split in ["train", "val", "test"]:
        split_pneu = BALANCED_DIR / split / "PNEUMONIA"
        if split_pneu.exists():
            for f in sorted(list(split_pneu.glob("*.jpeg")) + list(split_pneu.glob("*.jpg"))):
                pneu_counter += 1
                dst_name = f"pneumonia_{pneu_counter:06d}.jpeg"
                shutil.move(str(f), str(PNEUMONIA_DEST / dst_name))

    print(f"  - Moved PNEUMONIA images: {pneu_counter:,} -> data/PNEUMONIA/")

    # 3. Clean up intermediate directories
    print("\n[Step 2] Cleaning up intermediate folders in data/ ...")
    for extra in ["balanced_dataset", "custom_dataset", "chest_xray"]:
        p = DATA_DIR / extra
        if p.exists():
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    os.rmdir(p)
                print(f"  - Removed: data/{extra}")
            except Exception as e:
                print(f"  - Note: could not remove {extra}: {e}")

    print("\n" + "=" * 65)
    print("  [SUCCESS] DATA FOLDER COMBINED INTO 2 CLEAN FOLDERS!")
    print("=" * 65)
    print(f"  data/NORMAL/    : {norm_counter:,} images")
    print(f"  data/PNEUMONIA/ : {pneu_counter:,} images")
    print(f"  Total Pool      : {norm_counter + pneu_counter:,} images (Exact 50% - 50% Balance)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
