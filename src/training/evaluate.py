"""
src/training/evaluate.py
========================
Comprehensive evaluation of a trained model on the TEST set.

This is separate from the training loop because:
1. We evaluate only ONCE on the test set (at the very end).
2. We compute richer metrics (ROC curve, full confusion matrix, etc.).
3. We save all results and charts to the reports/ directory.

IMPORTANT: Never tune hyperparameters based on test-set performance.
The test set must remain "unseen" until the final evaluation.
Use the validation set for all decisions during training.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.metrics import compute_metrics, print_metrics
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)
from src.training.train import get_device


@torch.no_grad()
def evaluate_model(
    model:       nn.Module,
    test_loader: DataLoader,
    cfg:         dict,
    class_names: list[str] | None = None,
    model_name:  str = "Model",
    save_reports: bool = True,
) -> dict:
    """
    Evaluate the model on the test set and return a metrics dictionary.

    Parameters
    ----------
    model       : trained PyTorch model (already loaded with best weights)
    test_loader : DataLoader for the test set
    cfg         : config dictionary
    class_names : e.g. ["NORMAL", "PNEUMONIA"]
    model_name  : used in chart titles and filenames
    save_reports: if True, save confusion matrix and ROC curve to reports/

    Returns
    -------
    dict of all computed metrics
    """
    device      = get_device(cfg)
    model       = model.to(device)
    model.eval()
    class_names = class_names or cfg["dataset"]["classes"]
    reports_dir = Path(cfg["reports"]["output_dir"])

    all_labels = []   # true class indices
    all_preds  = []   # predicted class indices
    all_probs  = []   # probability of PNEUMONIA (class index 1)

    print(f"\n[Evaluate] Running evaluation on test set …")

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)                          # raw scores

        # Convert logits to probabilities using Softmax
        probs = torch.softmax(logits, dim=1)

        preds = probs.argmax(dim=1)                     # predicted class

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())    # prob of PNEUMONIA

    # ── Compute all metrics ──────────────────────────────────────────────
    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_prob=all_probs,
        class_names=class_names,
    )

    print_metrics(metrics, title=f"Test Evaluation — {model_name}")

    if save_reports:
        # ── Confusion matrix chart ──
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names=class_names,
            model_name=model_name,
            save_dir=reports_dir,
        )

        # ── ROC Curve ──
        if metrics["auc_roc"] is not None:
            from sklearn.metrics import roc_curve   # lazy: keeps sklearn/pandas out of the web server
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            plot_roc_curve(
                fpr, tpr,
                auc_score=metrics["auc_roc"],
                model_name=model_name,
                save_dir=reports_dir,
            )

    return metrics


def load_model_from_checkpoint(
    model:     nn.Module,
    ckpt_path: str | Path,
    device:    torch.device | None = None,
) -> nn.Module:
    """
    Load saved weights into a model.

    Parameters
    ----------
    model     : an empty (randomly initialised) model of the correct type
    ckpt_path : path to the .pth checkpoint file
    device    : target device (CPU/GPU)

    Returns
    -------
    model with loaded weights, moved to device
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Checkpoints saved by train.py store a dict; weights are under "model_state_dict"
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch      = checkpoint.get("epoch", "?")
        val_acc    = checkpoint.get("val_acc", "?")
        print(f"[Load] Checkpoint from epoch {epoch}, val_acc={val_acc:.4f}")
    else:
        # Fallback: the checkpoint IS the state dict
        state_dict = checkpoint

    # Strict on purpose: a mismatched checkpoint must fail loudly, not serve random weights
    model.load_state_dict(state_dict)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    print(f"[Load] Model loaded from: {ckpt_path}")
    return model
