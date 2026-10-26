"""
src/models/cnn.py
=================
Baseline Convolutional Neural Network (CNN) for chest X-ray classification.

This is a SIMPLE, CUSTOM network we build from scratch.  It won't match
ResNet-18's accuracy, but it gives us a useful baseline and a chance to
understand how CNNs work before using a pretrained model.

=== Beginner's Guide to CNN Concepts ===

CONVOLUTION (nn.Conv2d)
    Think of it like scanning a small "magnifying glass" (the filter/kernel)
    across the image.  Each filter learns to detect a specific pattern —
    edges, curves, textures.  After many filters we get a rich description
    of the image contents.

RELU ACTIVATION (nn.ReLU)
    After each convolution the network applies a simple rule: if the number
    is negative, replace it with 0; otherwise keep it.
    Formula: f(x) = max(0, x)
    Why?  Without a non-linear function, stacking convolutions would just
    be one big linear operation and couldn't learn complex patterns.

BATCH NORMALISATION (nn.BatchNorm2d)
    Normalises the output of each layer so values don't explode or vanish.
    Makes training faster and more stable.

MAX POOLING (nn.MaxPool2d)
    Shrinks the feature map by taking the MAXIMUM value in each small window.
    This makes the representation smaller, faster, and slightly translation-
    invariant (the model recognises the same pattern even if it's shifted a
    little).

DROPOUT (nn.Dropout)
    During training, randomly sets a fraction of neuron outputs to zero.
    This forces the network to learn redundant representations and prevents
    it from memorising training data (overfitting).

FULLY CONNECTED LAYER (nn.Linear)
    After convolutions, we flatten the feature maps into a 1-D vector and
    pass it through regular linear layers to produce the final class scores
    (called LOGITS).

LOGITS vs PROBABILITIES
    Logits are raw, unbounded numbers (e.g., [-1.2, 3.4]).
    Applying Softmax converts them to probabilities that sum to 1.0.
    CrossEntropyLoss applies Softmax internally, so our model outputs logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    A 3-block Convolutional Neural Network for binary chest X-ray classification.

    Architecture:
    ─────────────────────────────────────────────────────────────────
    Input:  (batch, 3, 224, 224)   — RGB image, 224×224 pixels
    Block 1: Conv(3→32) → BN → ReLU → MaxPool(2×2)   → (batch, 32, 112, 112)
    Block 2: Conv(32→64) → BN → ReLU → MaxPool(2×2)  → (batch, 64,  56,  56)
    Block 3: Conv(64→128) → BN → ReLU → MaxPool(2×2) → (batch, 128, 28,  28)
    AdaptiveAvgPool → (batch, 128, 1, 1) → Flatten → (batch, 128)
    Dropout(0.5)
    FC(128 → 64) → ReLU
    FC(64  → 2)  ← logits for 2 classes
    ─────────────────────────────────────────────────────────────────
    Total learnable parameters: ~200k  (much smaller than ResNet-18's ~11M)
    """

    def __init__(
        self,
        num_classes: int = 2,
        out_channels_1: int = 32,
        out_channels_2: int = 64,
        out_channels_3: int = 128,
        dropout: float = 0.5,
    ):
        """
        Parameters
        ----------
        num_classes    : number of output classes (2 for NORMAL/PNEUMONIA)
        out_channels_1 : number of filters in the first conv block
        out_channels_2 : number of filters in the second conv block
        out_channels_3 : number of filters in the third conv block
        dropout        : probability of dropping a neuron during training
        """
        super(BaselineCNN, self).__init__()

        # ── Block 1 ──────────────────────────────────────────────────────
        # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        #   in_channels=3  because RGB images have 3 channels (R, G, B)
        #   kernel_size=3  means a 3×3 filter
        #   padding=1      keeps the output the same spatial size as input
        self.block1 = nn.Sequential(
            nn.Conv2d(3, out_channels_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_1, out_channels_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # halve spatial dims
        )

        # ── Block 2 ──────────────────────────────────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_2, out_channels_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Block 3 ──────────────────────────────────────────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels_2, out_channels_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_3, out_channels_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Global Average Pooling ────────────────────────────────────────
        # Reduces (batch, 128, 28, 28) → (batch, 128, 1, 1)
        # This replaces a large Flatten + FC layer and reduces parameters.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classifier Head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # (batch, 128)
            nn.Dropout(p=dropout),
            nn.Linear(out_channels_3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),              # (batch, 2) — logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass: how data flows through the network.

        PyTorch automatically computes the backward pass (gradients)
        using autograd — we never need to write backward() ourselves.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, 3, 224, 224)

        Returns
        -------
        torch.Tensor, shape (batch_size, num_classes)
            Raw logits for each class.
        """
        x = self.block1(x)         # → (batch, 32, 112, 112)
        x = self.block2(x)         # → (batch, 64,  56,  56)
        x = self.block3(x)         # → (batch, 128, 28,  28)
        x = self.global_avg_pool(x)# → (batch, 128,  1,   1)
        x = self.classifier(x)     # → (batch, 2)
        return x

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_baseline_cnn(cfg: dict) -> BaselineCNN:
    """
    Instantiate the BaselineCNN using values from the config.

    Parameters
    ----------
    cfg : dict from get_config()

    Returns
    -------
    BaselineCNN model (on CPU by default; move to GPU with .to(device) later)
    """
    cnn_cfg = cfg["cnn"]
    model = BaselineCNN(
        num_classes    = len(cfg["dataset"]["classes"]),
        out_channels_1 = cnn_cfg["out_channels_1"],
        out_channels_2 = cnn_cfg["out_channels_2"],
        out_channels_3 = cnn_cfg["out_channels_3"],
        dropout        = cnn_cfg["dropout"],
    )
    print(f"\n[CNN] Baseline CNN created.")
    print(f"[CNN] Trainable parameters: {model.count_parameters():,}")
    return model


if __name__ == "__main__":
    # Quick sanity check — run with: python src/models/cnn.py
    import torch

    model  = BaselineCNN(num_classes=2)
    dummy  = torch.randn(4, 3, 224, 224)   # 4 fake images
    output = model(dummy)
    print(f"\nInput  shape: {dummy.shape}")
    print(f"Output shape: {output.shape}")   # expect (4, 2)
    print(f"Trainable parameters: {model.count_parameters():,}")
    print("Baseline CNN is working correctly ✓")
