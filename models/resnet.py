"""
src/models/resnet.py
====================
Transfer Learning with ResNet-18.

=== What is Transfer Learning? ===
Imagine learning to recognise X-ray abnormalities from scratch.
You would need to re-learn "what is a curve?", "what is an edge?", etc.
That would take millions of X-ray images.

Transfer learning says: "A model already trained on 1.2 million natural
images (ImageNet) already knows how to detect low-level features like
edges, textures, and shapes.  Let's REUSE those learned features and only
teach it the X-ray-specific decision at the end."

Result: much faster training, much better accuracy, with far fewer X-ray images.

=== ResNet-18 Architecture (brief) ===
ResNet stands for "Residual Network."
- Introduced by He et al. (2015) at Microsoft Research.
- The "18" means it has 18 layers with learnable weights.
- Key innovation: SKIP CONNECTIONS (residual connections) that add the
  input of a block directly to its output.
  Why?  Without skip connections, very deep networks suffer from "vanishing
  gradients" — gradients become so small during backpropagation that early
  layers stop learning.  Skip connections provide a "gradient highway"
  that keeps gradients flowing backward.

The final layer of ResNet-18 originally outputs 1000 class scores (ImageNet).
We REPLACE it with a 2-class layer for NORMAL vs PNEUMONIA.

=== Training Strategy ===
Phase A — Feature Extraction (frozen backbone):
    Only the new classification head trains.
    The backbone weights are frozen (not updated).
    This is fast and avoids destroying the pretrained features.

Phase B — Fine-Tuning (optional, unlocks later layers):
    After the head converges, we unfreeze the last few ResNet blocks and
    train everything together with a VERY SMALL learning rate.
    This allows the backbone to fine-tune its features specifically for X-rays.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class MedVisionResNet(nn.Module):
    """
    ResNet-18 adapted for binary chest X-ray classification.

    Parameters
    ----------
    num_classes   : int   — number of output classes (2)
    pretrained    : bool  — load ImageNet weights
    freeze_backbone: bool — if True, only the final FC layer trains initially
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super(MedVisionResNet, self).__init__()

        # ── Load the pretrained model ─────────────────────────────────────
        # weights=ResNet18_Weights.IMAGENET1K_V1 downloads the weights that
        # achieved the best accuracy on ImageNet with this architecture.
        if pretrained:
            print("[ResNet] Loading pretrained ImageNet weights …")
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            print("[ResNet] Initialising ResNet-18 with RANDOM weights …")
            self.backbone = models.resnet18(weights=None)

        # ── Replace the classifier head ────────────────────────────────────
        # The original ResNet-18 final layer: Linear(512, 1000)
        # We replace it with:                 Linear(512, 2)
        in_features = self.backbone.fc.in_features   # = 512 for ResNet-18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),                       # light regularisation
            nn.Linear(in_features, num_classes),     # 512 → 2
        )

        # ── Freeze the backbone if requested ──────────────────────────────
        if freeze_backbone:
            self.freeze_backbone()

    # ── Freeze / Unfreeze Utilities ───────────────────────────────────────
    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters except the final FC head.
        Frozen layers have requires_grad=False, so their weights are not
        updated during backpropagation.
        """
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:             # keep head unfrozen
                param.requires_grad = False
        print("[ResNet] Backbone FROZEN — only the classifier head will train.")

    def unfreeze_layers(self, n_layers: int = 2) -> None:
        """
        Unfreeze the last n_layers ResNet layer-groups for fine-tuning.

        ResNet-18 layer groups:   layer1, layer2, layer3, layer4
        n_layers=1 → unfreeze layer4
        n_layers=2 → unfreeze layer3 + layer4
        etc.

        We always keep fc unfrozen.
        """
        # Define the 4 layer groups + the final FC in order
        layer_names = ["layer1", "layer2", "layer3", "layer4", "fc"]
        to_unfreeze = set(layer_names[-(n_layers + 1):])  # +1 to always include fc

        unfrozen_params = 0
        for name, param in self.backbone.named_parameters():
            layer = name.split(".")[0]
            if layer in to_unfreeze:
                param.requires_grad = True
                unfrozen_params += param.numel()

        print(f"[ResNet] Fine-tuning: unfroze {n_layers} layer group(s). "
              f"Trainable parameter count now: {self.count_parameters():,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full ResNet + custom head.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 3, 224, 224)

        Returns
        -------
        torch.Tensor, shape (batch, num_classes)   — raw logits
        """
        return self.backbone(x)

    def count_parameters(self) -> int:
        """Return the number of currently TRAINABLE parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_feature_extractor(self) -> nn.Module:
        """
        Return the backbone WITHOUT the final FC layer.
        Used by Grad-CAM to hook into intermediate feature maps.
        """
        return nn.Sequential(*list(self.backbone.children())[:-1])  # all except fc

    def get_target_layer(self) -> nn.Module:
        """Return the layer used for Grad-CAM (last block of layer4)."""
        return self.backbone.layer4[-1]


def build_resnet(cfg: dict) -> MedVisionResNet:
    """
    Instantiate MedVisionResNet from config.

    Parameters
    ----------
    cfg : dict from get_config()

    Returns
    -------
    MedVisionResNet model
    """
    rcfg = cfg["resnet"]
    model = MedVisionResNet(
        num_classes     = len(cfg["dataset"]["classes"]),
        pretrained      = rcfg["pretrained"],
        freeze_backbone = rcfg["freeze_backbone"],
    )
    trainable = model.count_parameters()
    total     = sum(p.numel() for p in model.parameters())
    print(f"[ResNet] Total parameters    : {total:,}")
    print(f"[ResNet] Trainable parameters: {trainable:,}  "
          f"({100*trainable/total:.1f}% of total)")
    return model


if __name__ == "__main__":
    # Quick test: python src/models/resnet.py
    model  = MedVisionResNet(num_classes=2, pretrained=False)
    dummy  = torch.randn(4, 3, 224, 224)
    output = model(dummy)
    print(f"\nInput  shape: {dummy.shape}")
    print(f"Output shape: {output.shape}")   # expect (4, 2)
    print("ResNet-18 model is working correctly ✓")
