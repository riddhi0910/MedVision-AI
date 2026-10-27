"""
src/data/preprocessing.py
=========================
Image transforms and preprocessing pipelines for MedVision AI.

Includes:
- Training transforms (with safe medical augmentations: flip, slight rotation, color jitter)
- Validation / Test transforms (deterministic: resize + ImageNet normalization)
- Inference transforms (deterministic for single-image web app and CLI evaluation)
"""

from torchvision import transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224, mean=None, std=None):
    """
    Data augmentation and normalization pipeline for training.
    Applies medically valid transforms (horizontal flip, slight rotation +/- 10 deg).
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transform(image_size: int = 224, mean=None, std=None):
    """
    Deterministic preprocessing for validation and test sets.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_inference_transform(image_size: int = 224, mean=None, std=None):
    """
    Deterministic preprocessing for single-image inference in app.py.
    """
    return get_val_transform(image_size=image_size, mean=mean, std=std)


def denormalize(tensor, mean=None, std=None):
    """
    Reverse ImageNet normalization on a (C, H, W) PyTorch tensor.
    x_orig = x * std + mean
    """
    import torch
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

    return tensor * std_t + mean_t
