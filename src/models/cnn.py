"""
src/models/cnn.py
=================
Baseline Convolutional Neural Network (CNN) for chest X-ray classification.
Matches the trained Colab weights checkpoint (best_cnn.pth).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Custom 4-Block Convolutional Neural Network trained from scratch.
    Parameters: ~296k.
    """

    def __init__(
        self,
        num_classes: int = 2,
        out_channels_1: int = 32,
        out_channels_2: int = 64,
        out_channels_3: int = 128,
        dropout: float = 0.4,
        **kwargs
    ):
        super(BaselineCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, out_channels_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 2
            nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 3
            nn.Conv2d(out_channels_2, out_channels_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 4
            nn.Conv2d(out_channels_3, out_channels_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(out_channels_3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_baseline_cnn(cfg: dict) -> BaselineCNN:
    cnn_cfg = cfg.get("cnn", {})
    model = BaselineCNN(
        num_classes    = len(cfg.get("dataset", {}).get("classes", ["NORMAL", "PNEUMONIA"])),
        out_channels_1 = cnn_cfg.get("out_channels_1", 32),
        out_channels_2 = cnn_cfg.get("out_channels_2", 64),
        out_channels_3 = cnn_cfg.get("out_channels_3", 128),
        dropout        = cnn_cfg.get("dropout", 0.4),
    )
    print(f"\n[CNN] Baseline CNN created.")
    print(f"[CNN] Trainable parameters: {model.count_parameters():,}")
    return model
