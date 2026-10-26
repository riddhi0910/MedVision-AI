"""
src/training/train.py
=====================
The training loop — the core engine of the project.

=== How Training Works (Beginner Explanation) ===

One "EPOCH" = one full pass through the entire training dataset.

Within each epoch:
  1. Load a BATCH of images (e.g., 32 images).
  2. FORWARD PASS: Feed the batch through the model → get logits (predictions).
  3. COMPUTE LOSS: Compare predictions to true labels.
       CrossEntropyLoss measures how wrong the predictions are.
       Loss = 0 means perfect predictions; higher = more wrong.
  4. ZERO GRADIENTS: Clear old gradients from the previous step.
  5. BACKWARD PASS: PyTorch automatically computes how to adjust each weight
       to reduce the loss.  This uses the chain rule of calculus.
       We just call loss.backward() and PyTorch does all the maths.
  6. OPTIMIZER STEP: Update the model weights in the direction that reduces loss.
       optimizer.step() applies the weight updates.

After every epoch, we run a VALIDATION PASS (no gradient computation) to
check how well the model generalises to unseen data.

EARLY STOPPING: If validation loss doesn't improve for `patience` epochs,
we stop training.  This prevents overfitting.

CHECKPOINT SAVING: We save the model weights whenever validation accuracy
reaches a new best.
"""

import os
import time
import copy
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm   # progress bar for each batch

from src.utils.metrics import compute_metrics, print_metrics


# ── Device Helper ─────────────────────────────────────────────────────────────
def get_device(cfg: dict | None = None) -> torch.device:
    """
    Return the best available device (CUDA GPU → MPS Apple → CPU).

    The user can override with DEVICE in .env or config.
    """
    override = (cfg or {}).get("device_override", "").lower()
    if override in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if override == "cpu":
        return torch.device("cpu")
    # Auto-detect
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# ── Build Optimizer ────────────────────────────────────────────────────────────
def build_optimizer(model: nn.Module, cfg: dict, model_type: str = "resnet"):
    """
    Create an optimizer from the config.

    Optimizers control HOW weights are updated.
    - Adam  : adapts the learning rate per parameter.  Safe default.
    - AdamW : Adam + weight decay (L2 regularisation).  Slightly better
              for transformers and fine-tuning.
    """
    key = model_type  # "cnn" or "resnet"
    lr  = cfg[key]["learning_rate"]
    wd  = cfg[key].get("weight_decay", 0.0)
    opt = cfg[key].get("optimizer", "adam").lower()

    # Only pass parameters that require gradients
    params = filter(lambda p: p.requires_grad, model.parameters())

    if opt == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)


# ── Build Scheduler ────────────────────────────────────────────────────────────
def build_scheduler(optimizer, cfg: dict, model_type: str = "resnet"):
    """
    Learning-rate scheduler: reduce LR over time for better convergence.

    CosineAnnealingLR: starts at `lr`, smoothly decreases to near 0.
    StepLR:            multiply LR by 0.1 every `step_size` epochs.
    None:              constant LR.
    """
    sched_name = cfg[model_type].get("scheduler", "none").lower()
    epochs     = cfg[model_type]["epochs"]

    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif sched_name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    return None


# ── Single Epoch: Training ─────────────────────────────────────────────────────
def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Returns
    -------
    (avg_loss, accuracy)  over the entire training set for this epoch.
    """
    model.train()   # training mode: enables Dropout, BatchNorm updates
    running_loss  = 0.0
    correct       = 0
    total         = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False, ncols=80):
        images = images.to(device)
        labels = labels.to(device)

        # --- Zero gradients (MUST do this every step) ---
        optimizer.zero_grad()

        # --- Forward pass ---
        logits = model(images)             # raw class scores

        # --- Compute loss ---
        loss = criterion(logits, labels)   # CrossEntropyLoss

        # --- Backward pass (compute gradients) ---
        loss.backward()

        # --- Update weights ---
        optimizer.step()

        # --- Accumulate statistics ---
        running_loss += loss.item() * images.size(0)
        preds         = logits.argmax(dim=1)          # class with highest score
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ── Single Epoch: Validation ───────────────────────────────────────────────────
@torch.no_grad()   # disables gradient tracking — saves memory, speeds up eval
def validate_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Run one validation epoch (no weight updates).

    Returns
    -------
    (avg_loss, accuracy)  over the entire validation set.
    """
    model.eval()   # evaluation mode: disables Dropout, freezes BatchNorm stats
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ── Main Training Function ─────────────────────────────────────────────────────
def train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          dict,
    model_type:   str = "resnet",
    class_weights: torch.Tensor | None = None,
    save_name:    str = "best_model.pth",
) -> dict:
    """
    Full training loop with early stopping and best-model checkpointing.

    Parameters
    ----------
    model        : the PyTorch model to train
    train_loader : DataLoader for training data
    val_loader   : DataLoader for validation data
    cfg          : config dictionary
    model_type   : "cnn" or "resnet" (picks the right sub-config)
    class_weights: optional tensor for imbalanced classes
    save_name    : filename for the saved best model

    Returns
    -------
    dict  history with keys:
        "train_loss", "val_loss", "train_acc", "val_acc"
        (each is a list, one value per epoch)
    """
    device    = get_device(cfg)
    model     = model.to(device)
    epochs    = cfg[model_type]["epochs"]
    patience  = cfg["training"]["early_stopping_patience"]
    ckpt_dir  = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Train] Device      : {device}")
    print(f"[Train] Model type  : {model_type}")
    print(f"[Train] Epochs      : {epochs}")
    print(f"[Train] Batch size  : {cfg['dataloader']['batch_size']}")
    print(f"[Train] Patience    : {patience}")

    # ── Loss function ────────────────────────────────────────────────────
    # CrossEntropyLoss combines Softmax + NegativeLogLikelihood.
    # class_weights compensates for imbalance (PNEUMONIA >> NORMAL).
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"[Train] Using class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = build_optimizer(model, cfg, model_type)
    scheduler = build_scheduler(optimizer, cfg, model_type)

    # ── History ──────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    epochs_no_impr = 0
    best_weights   = copy.deepcopy(model.state_dict())

    print(f"\n{'─'*65}")
    print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
          f"{'Val Loss':>9}  {'Val Acc':>9}  {'LR':>8}")
    print(f"{'─'*65}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training ──
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        # ── Validation ──
        vl_loss, vl_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # ── LR Scheduler step ──
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # ── Record ──
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        # ── Log ──
        print(f"  {epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
              f"{vl_loss:>9.4f}  {vl_acc:>9.4f}  {current_lr:>8.6f}")

        # ── Best model checkpoint ──
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            best_val_acc   = vl_acc
            epochs_no_impr = 0
            best_weights   = copy.deepcopy(model.state_dict())
            save_path      = ckpt_dir / save_name
            torch.save({
                "epoch":      epoch,
                "model_state_dict": best_weights,
                "val_loss":   best_val_loss,
                "val_acc":    best_val_acc,
                "model_type": model_type,
            }, save_path)
            print(f"  ✓ New best — saved to {save_path}")
        else:
            epochs_no_impr += 1

        # ── Early Stopping ──
        if epochs_no_impr >= patience:
            print(f"\n[Train] Early stopping after {epoch} epochs "
                  f"(no improvement for {patience} epochs).")
            break

    elapsed = time.time() - start_time
    print(f"{'─'*65}")
    print(f"\n[Train] Training complete in {elapsed/60:.1f} min")
    print(f"[Train] Best val loss : {best_val_loss:.4f}")
    print(f"[Train] Best val acc  : {best_val_acc*100:.2f}%")

    # Restore the best weights into the model before returning
    model.load_state_dict(best_weights)
    history["best_val_acc"]  = best_val_acc
    history["best_val_loss"] = best_val_loss

    return history
