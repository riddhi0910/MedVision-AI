"""
src/utils/visualization.py
==========================
All plotting helpers live here.  We use Matplotlib and Seaborn.

Functions:
  - plot_class_distribution   : bar chart showing how many images per class
  - plot_sample_images        : grid showing random X-ray samples
  - plot_training_curves      : loss and accuracy vs. epoch
  - plot_confusion_matrix     : heatmap of the confusion matrix
  - plot_roc_curve            : ROC curve with AUC score
  - overlay_gradcam           : blend Grad-CAM heatmap onto the original image

All plots are saved to the reports/ directory AND optionally shown on-screen.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image

# Use a non-interactive backend so plots can be saved even without a display
# (important when running on a server or in background scripts).
import matplotlib
matplotlib.use("Agg")   # Must be set BEFORE importing pyplot in some environments


# ── Colour palette used throughout the project ──────────────────────────────
PALETTE = {
    "NORMAL":    "#4CAF84",   # teal-green
    "PNEUMONIA": "#E07B5A",   # coral-orange
    "accent":    "#5E8BCC",   # soft blue
    "dark":      "#2C3E50",
    "light":     "#F5F5F5",
}


def _ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return as Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── 1. Class Distribution Bar Chart ─────────────────────────────────────────
def plot_class_distribution(
    class_counts: dict,           # e.g. {"NORMAL": 1341, "PNEUMONIA": 3875}
    split_name: str = "Training", # label for the chart title
    save_dir: str | Path = "reports",
) -> str:
    """
    Draw a bar chart of image counts per class and save it.

    Returns the path to the saved figure.
    """
    save_dir = _ensure_dir(save_dir)
    classes = list(class_counts.keys())
    counts  = list(class_counts.values())
    colors  = [PALETTE.get(c, PALETTE["accent"]) for c in classes]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(classes, counts, color=colors, edgecolor="white", linewidth=1.5,
                  width=0.5)

    # Annotate each bar with the count
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{count:,}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color=PALETTE["dark"],
        )

    ax.set_title(f"Class Distribution — {split_name} Set",
                 fontsize=14, fontweight="bold", color=PALETTE["dark"])
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_facecolor(PALETTE["light"])
    fig.patch.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # Add percentage labels inside bars
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{pct:.1f}%",
            ha="center", va="center", fontsize=11, color="white", fontweight="bold",
        )

    plt.tight_layout()
    save_path = save_dir / f"class_dist_{split_name.lower()}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path}")
    return str(save_path)


# ── 2. Sample Image Grid ─────────────────────────────────────────────────────
def plot_sample_images(
    image_paths: dict,            # {"NORMAL": [path1, path2, ...], "PNEUMONIA": [...]}
    n_per_class: int = 4,
    save_dir: str | Path = "reports",
) -> str:
    """
    Display a grid of sample X-ray images, n_per_class from each class.

    Returns the path to the saved figure.
    """
    save_dir = _ensure_dir(save_dir)
    classes  = list(image_paths.keys())
    n_cols   = n_per_class
    n_rows   = len(classes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    if n_rows == 1:
        axes = [axes]  # ensure axes is always a 2-D list

    for row_idx, cls in enumerate(classes):
        paths = image_paths[cls][:n_cols]  # take at most n_cols images
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(paths):
                img = Image.open(paths[col_idx]).convert("RGB")
                ax.imshow(img, cmap="gray")
                if col_idx == 0:
                    ax.set_ylabel(cls, fontsize=12, fontweight="bold",
                                  color=PALETTE.get(cls, PALETTE["accent"]),
                                  rotation=90, labelpad=10)
            ax.axis("off")

    fig.suptitle("Sample Chest X-Ray Images", fontsize=15, fontweight="bold",
                 color=PALETTE["dark"], y=1.01)
    plt.tight_layout()
    save_path = save_dir / "sample_images.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path}")
    return str(save_path)


# ── 3. Training Curves ───────────────────────────────────────────────────────
def plot_training_curves(
    history: dict,             # {"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...]}
    model_name: str = "Model",
    save_dir: str | Path = "reports",
) -> str:
    """
    Plot training vs validation loss and accuracy over epochs.

    Returns the path to the saved figure.
    """
    save_dir = _ensure_dir(save_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss subplot ---
    ax_loss.plot(epochs, history["train_loss"], label="Train Loss",
                 color=PALETTE["accent"], linewidth=2, marker="o", markersize=4)
    ax_loss.plot(epochs, history["val_loss"],   label="Val Loss",
                 color=PALETTE["PNEUMONIA"], linewidth=2, marker="s", markersize=4,
                 linestyle="--")
    ax_loss.set_title(f"{model_name} — Loss", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(True, alpha=0.3)
    ax_loss.spines[["top", "right"]].set_visible(False)

    # --- Accuracy subplot ---
    ax_acc.plot(epochs, history["train_acc"], label="Train Accuracy",
                color=PALETTE["accent"], linewidth=2, marker="o", markersize=4)
    ax_acc.plot(epochs, history["val_acc"],   label="Val Accuracy",
                color=PALETTE["NORMAL"], linewidth=2, marker="s", markersize=4,
                linestyle="--")
    ax_acc.set_title(f"{model_name} — Accuracy", fontsize=13, fontweight="bold")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(); ax_acc.grid(True, alpha=0.3)
    ax_acc.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Training Curves — {model_name}", fontsize=14,
                 fontweight="bold", color=PALETTE["dark"])
    plt.tight_layout()
    save_path = save_dir / f"training_curves_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path}")
    return str(save_path)


# ── 4. Confusion Matrix Heatmap ──────────────────────────────────────────────
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] = None,
    model_name: str = "Model",
    save_dir: str | Path = "reports",
) -> str:
    """
    Draw a colour-coded confusion matrix and save it.

    Returns the path to the saved figure.
    """
    save_dir   = _ensure_dir(save_dir)
    class_names = class_names or ["NORMAL", "PNEUMONIA"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax, annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13,
                 fontweight="bold", color=PALETTE["dark"])
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    save_path = save_dir / f"confusion_matrix_{safe_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path}")
    return str(save_path)


# ── 5. ROC Curve ─────────────────────────────────────────────────────────────
def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    model_name: str = "Model",
    save_dir: str | Path = "reports",
) -> str:
    """
    Plot the Receiver Operating Characteristic curve.

    fpr : False-positive rate at each threshold
    tpr : True-positive rate  at each threshold
    auc_score : Area Under the Curve (0.5 = random, 1.0 = perfect)
    """
    save_dir = _ensure_dir(save_dir)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE["accent"], lw=2,
            label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1,
            label="Random Classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["accent"])
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold",
                 color=PALETTE["dark"])
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    save_path = save_dir / f"roc_curve_{safe_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path}")
    return str(save_path)


# ── 6. Grad-CAM Overlay ──────────────────────────────────────────────────────
def overlay_gradcam(
    original_image: np.ndarray,   # H×W×3 float32 in [0,1]
    heatmap: np.ndarray,          # H×W float32 in [0,1]  (Grad-CAM output)
    alpha: float = 0.4,
    colormap: str = "jet",
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Blend the Grad-CAM heatmap onto the original image.

    The heatmap is colourised and then transparently blended over the
    original grayscale/RGB X-ray so the radiologist (or student!) can
    see WHICH regions the model focused on.

    Parameters
    ----------
    original_image : np.ndarray, shape (H, W, 3), values in [0, 1]
    heatmap        : np.ndarray, shape (H, W),    values in [0, 1]
    alpha          : transparency of the heatmap overlay (0 = invisible, 1 = opaque)
    colormap       : matplotlib colormap name (default 'jet')
    save_path      : if given, saves the blended image to this path

    Returns
    -------
    np.ndarray  — blended image, shape (H, W, 3), values in [0, 255] uint8
    """
    import cv2

    # 1. Apply the chosen colormap to the 1-channel heatmap
    cmap      = plt.get_cmap(colormap)
    heatmap_c = cmap(heatmap)[:, :, :3]   # shape (H, W, 3), values in [0,1]

    # 2. Blend: blended = (1 - alpha) * original + alpha * heatmap
    original_image = np.clip(original_image, 0, 1)
    blended = (1 - alpha) * original_image + alpha * heatmap_c
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(blended).save(save_path)
        print(f"[Visualization] Grad-CAM overlay saved: {save_path}")

    return blended
