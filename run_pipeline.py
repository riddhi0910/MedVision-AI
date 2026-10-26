"""
run_pipeline.py — End-to-End Training Pipeline
===============================================
This script runs the COMPLETE project pipeline in order:

  Phase 2: Dataset exploration (counts, sample images, corruption check)
  Phase 3: Preprocessing verification
  Phase 4: Train Baseline CNN
  Phase 5: Train ResNet-18 (frozen backbone → then fine-tuning)
  Evaluate: Test-set evaluation for both models, save charts

Run from the MedVision-AI folder:
    python run_pipeline.py

Use Google Colab for GPU training — see notebooks/MedVision_AI.ipynb
"""

import sys
import json
import time
from pathlib import Path

# ── Make sure src/ is importable ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_config
from src.data.dataset import (
    find_dataset_root, print_dataset_summary, count_images,
    detect_corrupt_images, detect_duplicates_by_name,
    compute_class_weights, build_dataloaders,
)
from src.data.preprocessing import get_train_transform, get_val_transform
from src.models.cnn import build_baseline_cnn
from src.models.resnet import build_resnet
from src.training.train import train_model, get_device
from src.training.evaluate import evaluate_model
from src.utils.visualization import (
    plot_class_distribution, plot_sample_images, plot_training_curves
)


def run_phase2_exploration(cfg, dataset_root):
    """Phase 2: Explore the dataset."""
    print("\n" + "="*65)
    print("  PHASE 2: Dataset Exploration")
    print("="*65)

    # ── Summary table ──
    print_dataset_summary(dataset_root, cfg)

    # ── Class distribution charts ──
    reports_dir = Path(cfg["reports"]["output_dir"])
    for split in ["train", "val", "test"]:
        counts = count_images(dataset_root, split)
        if counts:
            plot_class_distribution(counts, split_name=split.capitalize(),
                                    save_dir=reports_dir)

    # ── Sample images grid ──
    train_dir  = dataset_root / "train"
    classes    = cfg["dataset"]["classes"]
    img_paths  = {}
    for cls in classes:
        cls_dir = train_dir / cls
        if cls_dir.exists():
            paths = sorted(cls_dir.glob("*.jpeg")) + sorted(cls_dir.glob("*.jpg"))
            img_paths[cls] = [str(p) for p in paths[:8]]   # up to 8 samples

    if img_paths:
        plot_sample_images(img_paths, n_per_class=4, save_dir=reports_dir)

    # ── Corrupt image check ──
    print("\n[Phase 2] Checking for corrupt images …")
    corrupt = detect_corrupt_images(dataset_root, splits=["train", "test"])
    if corrupt:
        print(f"  ⚠️ Found {len(corrupt)} corrupt files:")
        for p in corrupt[:10]:
            print(f"    {p}")
    else:
        print("  ✓ No corrupt images found.")

    # ── Duplicate filename check ──
    print("\n[Phase 2] Checking for duplicate filenames …")
    dupes = detect_duplicates_by_name(dataset_root)

    print(f"\n[Phase 2] Complete. Charts saved to: {reports_dir}")


def run_phase4_cnn(cfg, train_loader, val_loader, test_loader,
                   class_names, class_weights, reports_dir):
    """Phase 4: Train the Baseline CNN."""
    print("\n" + "="*65)
    print("  PHASE 4: Baseline CNN Training")
    print("="*65)

    model = build_baseline_cnn(cfg)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        model_type="cnn",
        class_weights=class_weights if cfg["training"]["use_class_weights"] else None,
        save_name="best_cnn.pth",
    )

    # Save training curves
    plot_training_curves(history, model_name="Baseline CNN", save_dir=reports_dir)

    # Test evaluation
    print("\n[Phase 4] Evaluating CNN on test set …")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        class_names=class_names,
        model_name="Baseline CNN",
    )

    # Save metrics to JSON
    metrics_path = reports_dir / "cnn_metrics.json"
    save_metrics = {k: v for k, v in metrics.items()
                    if k not in ("confusion_matrix", "classification_report")}
    save_metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"[Phase 4] Metrics saved to: {metrics_path}")

    return model, history, metrics


def run_phase5_resnet(cfg, train_loader, val_loader, test_loader,
                      class_names, class_weights, reports_dir):
    """Phase 5: Train ResNet-18 with transfer learning."""
    print("\n" + "="*65)
    print("  PHASE 5A: ResNet-18 — Frozen Backbone (Head Only)")
    print("="*65)

    model = build_resnet(cfg)

    # ── Phase 5A: train the head only ──
    cfg_5a       = dict(cfg)
    cfg_5a["resnet"] = dict(cfg["resnet"])
    cfg_5a["resnet"]["epochs"] = min(10, cfg["resnet"]["epochs"])  # quick head training

    history_a = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg_5a,
        model_type="resnet",
        class_weights=class_weights if cfg["training"]["use_class_weights"] else None,
        save_name="best_resnet18_headonly.pth",
    )

    # ── Phase 5B: fine-tune last layers ──
    print("\n" + "="*65)
    print("  PHASE 5B: ResNet-18 — Fine-Tuning (Unfreezing Last Layers)")
    print("="*65)

    n_unfreeze = cfg["resnet"].get("fine_tune_layers", 2)
    model.unfreeze_layers(n_unfreeze)

    # Use a much smaller learning rate for fine-tuning
    import torch.optim
    from src.training.train import build_optimizer, build_scheduler
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["resnet"]["learning_rate"] * 0.1,   # 10x smaller
        weight_decay=cfg["resnet"]["weight_decay"],
    )
    # We'll use the full training function for simplicity
    history_b = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        model_type="resnet",
        class_weights=class_weights if cfg["training"]["use_class_weights"] else None,
        save_name="best_resnet18.pth",
    )

    # Merge histories for combined plot
    combined_history = {
        "train_loss": history_a["train_loss"] + history_b["train_loss"],
        "val_loss":   history_a["val_loss"]   + history_b["val_loss"],
        "train_acc":  history_a["train_acc"]  + history_b["train_acc"],
        "val_acc":    history_a["val_acc"]    + history_b["val_acc"],
    }
    plot_training_curves(combined_history, model_name="ResNet-18",
                         save_dir=reports_dir)

    # Test evaluation
    print("\n[Phase 5] Evaluating ResNet-18 on test set …")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        class_names=class_names,
        model_name="ResNet-18",
    )

    metrics_path = reports_dir / "resnet18_metrics.json"
    save_metrics = {k: v for k, v in metrics.items()
                    if k not in ("confusion_matrix", "classification_report")}
    save_metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"[Phase 5] Metrics saved to: {metrics_path}")

    return model, combined_history, metrics


def main():
    print("\n" + "█"*65)
    print("  MedVision AI — Full Training Pipeline")
    print("█"*65)

    # ── Load config ──
    cfg         = get_config()
    reports_dir = Path(cfg["reports"]["output_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    device      = get_device(cfg)
    print(f"\n[Pipeline] Using device: {device}")

    # ── Find dataset ──
    dataset_root = find_dataset_root(cfg)

    # ── Phase 2: Exploration ──
    run_phase2_exploration(cfg, dataset_root)

    # ── Phase 3: Build DataLoaders ──
    print("\n" + "="*65)
    print("  PHASE 3: Building DataLoaders")
    print("="*65)
    image_size    = cfg["preprocessing"]["image_size"]
    train_tf      = get_train_transform(image_size)
    val_tf        = get_val_transform(image_size)
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        cfg=cfg,
        train_transform=train_tf,
        val_transform=val_tf,
        dataset_root=dataset_root,
    )
    print(f"[Phase 3] Class names: {class_names}")

    # ── Class weights ──
    class_weights = compute_class_weights(dataset_root, cfg)

    # ── Phase 4: Baseline CNN ──
    run_phase4_cnn(cfg, train_loader, val_loader, test_loader,
                   class_names, class_weights, reports_dir)

    # ── Phase 5: ResNet-18 ──
    run_phase5_resnet(cfg, train_loader, val_loader, test_loader,
                      class_names, class_weights, reports_dir)

    print("\n" + "█"*65)
    print("  Pipeline complete!  Reports saved to:", reports_dir)
    print("  Run the web app with:  streamlit run app.py")
    print("█"*65)


if __name__ == "__main__":
    main()
