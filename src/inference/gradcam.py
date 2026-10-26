"""
src/inference/gradcam.py / backend/inference/gradcam.py
=======================================================
Grad-CAM: Gradient-weighted Class Activation Mapping.
Native PyTorch Implementation (Self-contained, Robust & Viva-Ready).

=== How it works (Selvaraju et al., ICCV 2017) ===
1. Forward Hook: Captures feature map activations A^k at the target layer.
2. Backward Hook: Captures gradients d(y^c)/dA^k of target class score y^c.
3. Global Average Pooling: alpha_k = (1/Z) * sum(gradients).
4. Weighted sum: L_GradCAM = ReLU( sum(alpha_k * A^k) ).
5. Bilinear upsampling to input image dimensions (224, 224) & normalization to [0, 1].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.preprocessing import denormalize
from src.utils.visualization import overlay_gradcam


def generate_gradcam(
    model:        nn.Module,
    input_tensor: torch.Tensor,      # shape (1, 3, 224, 224)
    target_layer: nn.Module,
    target_class: int | None = None,  # None = use the predicted class
    use_cuda:     bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a Grad-CAM heatmap for one input image using native PyTorch hooks.

    Parameters
    ----------
    model        : trained model
    input_tensor : preprocessed image tensor, shape (1, 3, H, W)
    target_layer : convolutional layer to generate CAM for (e.g. layer4[-1])
    target_class : class index to explain (default: predicted class)

    Returns
    -------
    (heatmap, original_image_array)
    heatmap              : np.ndarray shape (H, W), values in [0, 1]
    original_image_array : np.ndarray shape (H, W, 3), values in [0, 1]
    """
    model.eval()

    activations = []
    gradients   = []

    def forward_hook(module, input_act, output_act):
        activations.append(output_act)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register forward and backward hooks on the target convolutional layer
    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        # 1. Forward pass
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # 2. Backward pass on target class score
        model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if len(activations) == 0 or len(gradients) == 0:
            raise RuntimeError("Grad-CAM hooks failed to capture feature maps or gradients.")

        # 3. Compute weights: Global Average Pooling on gradients
        act  = activations[0].detach()   # (1, C, H_feat, W_feat)
        grad = gradients[0].detach()      # (1, C, H_feat, W_feat)

        alpha = torch.mean(grad, dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # 4. Weighted combination & ReLU
        cam = torch.sum(alpha * act, dim=1, keepdim=True)    # (1, 1, H_feat, W_feat)
        cam = F.relu(cam)

        # 5. Upsample to original image resolution
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h_in, w_in), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # 6. Normalize to [0, 1]
        c_min, c_max = float(cam.min()), float(cam.max())
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        # 7. Recover normalized input to RGB displayable numpy image
        orig_tensor = denormalize(input_tensor[0].detach().cpu())
        orig_img = orig_tensor.permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1).astype(np.float32)

        return cam.astype(np.float32), orig_img

    finally:
        # Always remove hooks to prevent memory leaks
        h_fwd.remove()
        h_bwd.remove()


def generate_and_save_gradcam(
    model:        nn.Module,
    input_tensor:  torch.Tensor,
    target_layer:  nn.Module,
    save_path:     str,
    target_class:  int | None = None,
    alpha:         float = 0.4,
    use_cuda:      bool = False,
) -> np.ndarray:
    """Generate Grad-CAM, overlay it on the original image, and save to file."""
    heatmap, original_image = generate_gradcam(
        model, input_tensor, target_layer, target_class, use_cuda
    )

    blended = overlay_gradcam(
        original_image=original_image,
        heatmap=heatmap,
        alpha=alpha,
        save_path=save_path,
    )
    return blended
