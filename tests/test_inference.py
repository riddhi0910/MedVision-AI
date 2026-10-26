"""
tests/test_inference.py
=======================
Unit tests for the inference pipeline.

Run with:  pytest tests/test_inference.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io
import numpy as np
import torch
import pytest
from PIL import Image
from src.inference.predict import predict_single_image
from src.models.cnn import BaselineCNN


@pytest.fixture
def dummy_xray():
    """Create a fake grayscale-looking PIL image (224×224)."""
    arr = np.random.randint(30, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def dummy_model():
    """Untrained CNN for testing inference code paths."""
    model = BaselineCNN(num_classes=2)
    model.eval()
    return model


@pytest.fixture
def minimal_cfg():
    return {
        "dataset":       {"classes": ["NORMAL", "PNEUMONIA"]},
        "preprocessing": {"image_size": 224},
        "device_override": "cpu",
    }


class TestPredictSingleImage:

    def test_returns_dict_keys(self, dummy_xray, dummy_model, minimal_cfg):
        result = predict_single_image(
            image=dummy_xray,
            model=dummy_model,
            cfg=minimal_cfg,
            generate_cam=False,   # skip CAM in unit test (no trained weights)
        )
        required_keys = {
            "predicted_class", "predicted_index", "confidence",
            "class_scores", "is_uncertain",
        }
        assert required_keys.issubset(result.keys()), \
            f"Missing keys: {required_keys - result.keys()}"

    def test_predicted_class_is_valid(self, dummy_xray, dummy_model, minimal_cfg):
        result = predict_single_image(dummy_xray, dummy_model, minimal_cfg,
                                       generate_cam=False)
        assert result["predicted_class"] in ["NORMAL", "PNEUMONIA"]

    def test_confidence_in_range(self, dummy_xray, dummy_model, minimal_cfg):
        result = predict_single_image(dummy_xray, dummy_model, minimal_cfg,
                                       generate_cam=False)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_class_scores_sum_to_one(self, dummy_xray, dummy_model, minimal_cfg):
        result = predict_single_image(dummy_xray, dummy_model, minimal_cfg,
                                       generate_cam=False)
        total = sum(result["class_scores"].values())
        assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected 1.0"

    def test_accepts_file_bytes(self, tmp_path, dummy_model, minimal_cfg):
        """Test that inference works from a file path."""
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        result = predict_single_image(img_path, dummy_model, minimal_cfg,
                                       generate_cam=False)
        assert result["predicted_class"] in ["NORMAL", "PNEUMONIA"]
