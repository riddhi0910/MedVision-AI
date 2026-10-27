"""
src/data/dataset.py
===================
Dataset loading, validation, and DataLoader constructors for Chest X-Ray images.
Supports Kaggle/Kermany et al. dataset structure (train / val / test splits with NORMAL / PNEUMONIA).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image


def find_dataset_root(cfg: dict) -> Path:
    """Find and return valid dataset root directory."""
    env_root = os.getenv("DATASET_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)

    cfg_path = Path(cfg["dataset"]["root"])
    if cfg_path.exists():
        return cfg_path

    # Try common fallback relative paths
    candidates = [
        Path("data/chest_xray"),
        Path("data/chest_xray/chest_xray"),
        Path("../data/chest_xray"),
    ]
    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c

    return cfg_path


def count_images(dataset_root: Path, split: str) -> Dict[str, int]:
    """Count number of images per class in a split."""
    split_dir = dataset_root / split
    counts = {}
    if not split_dir.exists():
        return counts

    for cls_dir in sorted(split_dir.iterdir()):
        if cls_dir.is_dir() and not cls_dir.name.startswith("."):
            exts = ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG")
            total = sum(len(list(cls_dir.glob(ext))) for ext in exts)
            counts[cls_dir.name] = total
    return counts


def print_dataset_summary(dataset_root: Path, cfg: dict) -> None:
    """Print tabular summary of image counts across train, val, test or direct classes."""
    print("\n[Dataset] Summary:")
    print("-" * 50)

    # Check if pre-split or combined
    if (dataset_root / "NORMAL").exists() and (dataset_root / "PNEUMONIA").exists():
        norm_count = len(list((dataset_root / "NORMAL").glob("*.jpeg"))) + len(list((dataset_root / "NORMAL").glob("*.jpg")))
        pneu_count = len(list((dataset_root / "PNEUMONIA").glob("*.jpeg"))) + len(list((dataset_root / "PNEUMONIA").glob("*.jpg")))
        total = norm_count + pneu_count
        print(f"Combined 2-Class Pool at: {dataset_root}")
        print(f"  NORMAL    : {norm_count:,} images ({norm_count/total*100:.1f}%)")
        print(f"  PNEUMONIA : {pneu_count:,} images ({pneu_count/total*100:.1f}%)")
        print(f"  Total Pool: {total:,} images (Exact 50-50 Balance)")
        print("-" * 50 + "\n")
        return

    print(f"{'Split':<12} {'NORMAL':<12} {'PNEUMONIA':<12} {'Total':<10}")
    print("-" * 50)

    grand_total = 0
    for split in ["train", "val", "test"]:
        counts = count_images(dataset_root, split)
        norm = counts.get("NORMAL", 0)
        pneu = counts.get("PNEUMONIA", 0)
        total = norm + pneu
        grand_total += total
        print(f"{split:<12} {norm:<12} {pneu:<12} {total:<10}")
    print("-" * 50)
    print(f"{'Total':<12} {'':<12} {'':<12} {grand_total:<10}\n")


def detect_corrupt_images(dataset_root: Path, splits: Optional[List[str]] = None) -> List[Path]:
    """Verify that all image files can be loaded by PIL without errors."""
    splits = splits or ["train", "val", "test"]
    corrupt = []
    exts = ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG")

    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for ext in exts:
            for img_path in split_dir.rglob(ext):
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    corrupt.append(img_path)
    return corrupt


def detect_duplicates_by_name(dataset_root: Path) -> Dict[str, List[Path]]:
    """Check if identical filenames exist across splits."""
    files_by_name: Dict[str, List[Path]] = {}
    exts = ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG")

    for ext in exts:
        for img_path in dataset_root.rglob(ext):
            files_by_name.setdefault(img_path.name, []).append(img_path)

    dupes = {name: paths for name, paths in files_by_name.items() if len(paths) > 1}
    if dupes:
        print(f"  [Info] {len(dupes)} filename collision(s) detected across splits.")
    else:
        print("  ✓ No duplicate filenames found across splits.")
    return dupes


def compute_class_weights(dataset_root: Path, cfg: dict) -> torch.Tensor:
    """
    Calculate inverse class frequency weights to handle clinical class imbalance.
    NORMAL (~27%) gets higher weight than PNEUMONIA (~73%).
    """
    train_counts = count_images(dataset_root, "train")
    classes = cfg["dataset"]["classes"]
    counts = [train_counts.get(cls, 1) for cls in classes]

    total = sum(counts)
    # Inverse frequency weights: w_i = total / (num_classes * count_i)
    weights = [total / (len(classes) * c) if c > 0 else 1.0 for c in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print(f"[Dataset] Computed Class Weights: {dict(zip(classes, [round(w, 3) for w in weights]))}")
    return weights_tensor


class TransformSubset(Dataset):
    """Subset wrapper that applies a specific transform to the underlying dataset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            # If img is a PIL image, apply transform
            img = self.transform(img)
        return img, label


def build_dataloaders(
    cfg: dict,
    train_transform,
    val_transform,
    dataset_root: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Construct train, val, and test PyTorch DataLoaders.
    Supports:
    1. Pre-split directories (train/, val/, test/)
    2. Combined class folders (NORMAL/, PNEUMONIA/) with 80/10/10 random split.
    """
    dataset_root = dataset_root or find_dataset_root(cfg)
    batch_size = cfg["dataloader"]["batch_size"]
    num_workers = cfg["dataloader"].get("num_workers", 0)
    pin_memory = cfg["dataloader"].get("pin_memory", False) and torch.cuda.is_available()

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    # Case 1: Pre-split subfolders exist
    if train_dir.exists() and (train_dir / "NORMAL").exists():
        train_dataset = ImageFolder(str(train_dir), transform=train_transform)
        val_dataset = ImageFolder(str(val_dir), transform=val_transform)
        test_dataset = ImageFolder(str(test_dir), transform=val_transform)
        class_names = train_dataset.classes
    else:
        # Case 2: Combined 2-folder layout (NORMAL, PNEUMONIA directly in dataset_root)
        print(f"[DataLoader] Using combined 2-class pool at: {dataset_root}")
        raw_dataset = ImageFolder(str(dataset_root))
        class_names = raw_dataset.classes
        total_len = len(raw_dataset)

        seed = cfg["dataset"].get("random_seed", 42)
        n_train = int(0.80 * total_len)
        n_val = int(0.10 * total_len)
        n_test = total_len - n_train - n_val

        generator = torch.Generator().manual_seed(seed)
        sub_train, sub_val, sub_test = torch.utils.data.random_split(
            raw_dataset, [n_train, n_val, n_test], generator=generator
        )

        train_dataset = TransformSubset(sub_train, train_transform)
        val_dataset = TransformSubset(sub_val, val_transform)
        test_dataset = TransformSubset(sub_test, val_transform)
        print(f"[DataLoader] Random Split 80/10/10 -> Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_names
