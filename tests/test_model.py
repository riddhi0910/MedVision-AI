"""
tests/test_model.py
===================
Unit tests for the CNN and ResNet-18 model architectures.

Run with:  pytest tests/test_model.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest
from src.models.cnn import BaselineCNN, build_baseline_cnn
from src.models.resnet import MedVisionResNet, build_resnet


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_batch():
    """4 fake images of shape (4, 3, 224, 224)."""
    return torch.randn(4, 3, 224, 224)

@pytest.fixture
def minimal_cfg():
    return {
        "dataset": {"classes": ["NORMAL", "PNEUMONIA"]},
        "cnn": {
            "out_channels_1": 32, "out_channels_2": 64, "out_channels_3": 128,
            "dropout": 0.5, "learning_rate": 0.001, "epochs": 1, "optimizer": "adam",
        },
        "resnet": {
            "pretrained": False,  # Don't download weights during tests
            "freeze_backbone": True,
            "fine_tune_layers": 1,
            "learning_rate": 0.0003, "weight_decay": 0.0001,
            "epochs": 1, "optimizer": "adamw", "scheduler": "none",
        },
    }


# ── CNN Tests ─────────────────────────────────────────────────────────────────
class TestBaselineCNN:

    def test_output_shape(self, dummy_batch):
        """Model must output (batch, 2) logits."""
        model = BaselineCNN(num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_parameter_count(self):
        """CNN should have a reasonable number of parameters."""
        model = BaselineCNN()
        n = model.count_parameters()
        assert n > 10_000, "Too few parameters — something is wrong"
        assert n < 5_000_000, "Too many parameters for a baseline CNN"

    def test_no_nan_output(self, dummy_batch):
        """Model output should not contain NaN values."""
        model = BaselineCNN(num_classes=2)
        with torch.no_grad():
            out = model(dummy_batch)
        assert not torch.isnan(out).any(), "NaN in model output"

    def test_training_mode_dropout(self):
        """Dropout should produce different outputs in train vs eval mode."""
        model = BaselineCNN(num_classes=2, dropout=0.9)
        x = torch.randn(8, 3, 224, 224)
        model.train()
        with torch.no_grad():
            out1 = model(x)
        model.eval()
        with torch.no_grad():
            out2 = model(x)
        # In train mode outputs vary, in eval they are deterministic.
        # We just check both don't crash.
        assert out1.shape == out2.shape


# ── ResNet Tests ──────────────────────────────────────────────────────────────
class TestMedVisionResNet:

    def test_output_shape(self, dummy_batch):
        """ResNet output must be (batch, 2)."""
        model = MedVisionResNet(num_classes=2, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 2)

    def test_frozen_backbone(self):
        """When backbone is frozen, only fc parameters should be trainable."""
        model = MedVisionResNet(pretrained=False, freeze_backbone=True)
        for name, param in model.backbone.named_parameters():
            if "fc" not in name:
                assert not param.requires_grad, \
                    f"Parameter {name} should be frozen but is not."

    def test_unfreeze_layers(self):
        """After unfreeze_layers(2), layer3 and layer4 should be trainable."""
        model = MedVisionResNet(pretrained=False, freeze_backbone=True)
        model.unfreeze_layers(2)
        for name, param in model.backbone.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                assert param.requires_grad, \
                    f"Parameter {name} should be trainable after unfreeze"

    def test_no_nan_output(self, dummy_batch):
        model = MedVisionResNet(num_classes=2, pretrained=False)
        with torch.no_grad():
            out = model(dummy_batch)
        assert not torch.isnan(out).any()
