"""
src/utils/metrics.py
====================
All evaluation metrics live here so they can be reused by both the
training loop and the final test evaluation without code duplication.

Metrics we compute:
  - Accuracy          : fraction of correct predictions
  - Precision         : of all "PNEUMONIA" predictions, how many were right?
  - Recall / Sensitivity: of all actual PNEUMONIA cases, how many did we catch?
  - F1 Score          : harmonic mean of precision and recall
  - Specificity       : of all NORMAL cases, how many did we correctly label NORMAL?
  - AUC-ROC           : Area Under the ROC Curve — threshold-independent quality measure
  - Confusion Matrix  : 2×2 table of TP, FP, TN, FN

Why so many metrics?
  In medical imaging, raw accuracy is misleading.  If 73% of images are
  PNEUMONIA, a model that always predicts PNEUMONIA gets 73% accuracy —
  but it never detects NORMAL patients correctly.  Recall and specificity
  together give a complete picture of clinical usefulness.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_prob: list | np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth class indices (0 = NORMAL, 1 = PNEUMONIA).
    y_pred : array-like of int
        Predicted class indices from the model.
    y_prob : array-like of float, optional
        Predicted probability for the POSITIVE class (PNEUMONIA).
        Required for AUC-ROC.
    class_names : list of str, optional
        Human-readable names for each class.

    Returns
    -------
    dict
        Dictionary with all metric values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    metrics = {}

    # --- Basic Metrics ------------------------------------------------
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # zero_division=0 means: if a class has no predictions, set metric to 0
    # rather than crashing with a division-by-zero error.
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"]    = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"]  = f1_score(y_true, y_pred, zero_division=0)

    # --- Confusion Matrix ---------------------------------------------
    # Layout:
    #               Predicted NORMAL   Predicted PNEUMONIA
    # Actual NORMAL       TN                  FP
    # Actual PNEUMON      FN                  TP
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        # Specificity = TN / (TN + FP)  — how well we detect NORMAL
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["true_positives"]  = int(tp)
        metrics["true_negatives"]  = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

    # --- AUC-ROC (needs probability scores, not just hard predictions) -
    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Can fail if only one class is present in y_true
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None

    # --- Full Classification Report (as string, for logging) ----------
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Results") -> None:
    """Pretty-print the metrics dictionary to the console."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  F1 Score    : {metrics['f1_score']:.4f}")
    if "specificity" in metrics:
        print(f"  Specificity : {metrics['specificity']:.4f}")
    if metrics.get("auc_roc") is not None:
        print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")
    print(f"\n  Classification Report:")
    print(metrics["classification_report"])
    print(f"{'='*55}\n")
