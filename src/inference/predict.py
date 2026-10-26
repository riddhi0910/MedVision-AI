"""
src/inference/predict.py
========================
Single-image inference pipeline for the Streamlit web app and CLI use.

This module:
1. Takes a PIL Image or file path.
2. Applies the inference transform (resize + normalise).
3. Runs the model forward pass.
4. Returns predicted class, confidence scores, and (optionally) Grad-CAM.

Designed to be imported by app.py.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.data.preprocessing import get_inference_transform
from src.inference.gradcam import generate_gradcam
from src.training.train import get_device
from src.utils.config_loader import get_config


# ── Confidence Threshold ─────────────────────────────────────────────────────
# If the highest probability is below this threshold, the model is uncertain.
UNCERTAINTY_THRESHOLD = 0.60   # 60%


def predict_single_image(
    image:        Union[str, Path, Image.Image],
    model:        nn.Module,
    cfg:          dict,
    class_names:  list[str] | None = None,
    generate_cam: bool = True,
) -> dict:
    """
    Run inference on a single chest X-ray image.

    Parameters
    ----------
    image        : file path or PIL Image object
    model        : trained and loaded PyTorch model
    cfg          : config dictionary
    class_names  : list like ["NORMAL", "PNEUMONIA"]
    generate_cam : whether to compute Grad-CAM heatmap

    Returns
    -------
    dict with keys:
      "predicted_class"   : str   — "NORMAL" or "PNEUMONIA"
      "predicted_index"   : int   — 0 or 1
      "confidence"        : float — probability of predicted class (0–1)
      "class_scores"      : dict  — {class_name: probability}
      "is_uncertain"      : bool  — True if confidence < threshold
      "heatmap"           : np.ndarray or None  — Grad-CAM heatmap (H,W)
      "original_image"    : np.ndarray          — denormalised image (H,W,3)
    """
    device       = get_device(cfg)
    class_names  = class_names or cfg["dataset"]["classes"]
    image_size   = cfg["preprocessing"]["image_size"]
    transform    = get_inference_transform(image_size)

    # ── 1. Load image ────────────────────────────────────────────────────
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # ── 2. Preprocess ────────────────────────────────────────────────────
    input_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # ── 3. Forward pass ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)                         # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)[0]            # (num_classes,)

    probs_list = probs.cpu().numpy().tolist()
    pred_idx   = int(np.argmax(probs_list))
    confidence = float(probs_list[pred_idx])

    class_scores = {name: float(p) for name, p in zip(class_names, probs_list)}

    # ── 4. Grad-CAM ──────────────────────────────────────────────────────
    heatmap       = None
    original_img  = None

    if generate_cam:
        try:
            # Get the target layer for Grad-CAM
            # Works for MedVisionResNet; BaselineCNN uses block3
            if hasattr(model, "backbone"):
                target_layer = model.get_target_layer()    # ResNet layer4
            else:
                target_layer = model.block3[-2]            # Last Conv in CNN

            use_cuda = (device.type == "cuda")
            heatmap, original_img = generate_gradcam(
                model=model,
                input_tensor=input_tensor,
                target_layer=target_layer,
                target_class=pred_idx,
                use_cuda=use_cuda,
            )
        except Exception as e:
            print(f"[Predict] Grad-CAM failed: {e}. Continuing without heatmap.")

    return {
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "confidence":      confidence,
        "class_scores":    class_scores,
        "is_uncertain":    confidence < UNCERTAINTY_THRESHOLD,
        "heatmap":         heatmap,
        "original_image":  original_img,
    }
