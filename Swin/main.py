"""
EEEM068 — DR Grading
main.py — Main experiment runner.

Run this to train both models and compare them.
Designed for Google Colab / Surrey AI Supercomputer.

EXPERIMENT LOG FORMAT (upload to GitHub weekly):
  Each run logs: model, strategy, augment_level, all metrics.
  This satisfies the "weekly commits with training logs" requirement.
"""

import os
import json
import torch
import numpy as np
import random

from config import cfg
from dataset import build_dataloaders
from models import build_swin_model, build_efficientnet_model
from trainer import Trainer
from metrics import MetricTracker, print_metrics, plot_confusion_matrix
from gradcam import GradCAM, visualise_gradcam_batch


# ─── Reproducibility ────────────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(cfg.SEED)


# ─── Mount Google Drive (Colab only) ────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')


# ─── Install dependencies ────────────────────────────────────────────────────
# !pip install timm albumentations -q


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Swin Transformer (your model)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXPERIMENT 1 — Swin-Base + weighted_sampler + full augment")
print("="*60)

cfg.IMBALANCE_STRATEGY = "weighted_sampler"

train_loader, val_loader, test_loader, class_weights = build_dataloaders(
    image_dir     = cfg.DATA_DIR + "/train",
    csv_path      = cfg.CSV_PATH,
    strategy      = cfg.IMBALANCE_STRATEGY,
    augment_level = "full",
)

swin_model = build_swin_model(pretrained=True)

swin_trainer = Trainer(
    model            = swin_model,
    train_loader     = train_loader,
    val_loader       = val_loader,
    class_weights    = class_weights,
    model_name       = "swin_base",
    use_class_weight = False,   # using sampler instead
)

swin_history = swin_trainer.train(epochs=cfg.EPOCHS)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: EfficientNet-B4 baseline
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXPERIMENT 2 — EfficientNet-B4 + class_weight + full augment")
print("="*60)

cfg.IMBALANCE_STRATEGY = "class_weight"

train_loader2, val_loader2, test_loader2, class_weights2 = build_dataloaders(
    image_dir     = cfg.DATA_DIR + "/train",
    csv_path      = cfg.CSV_PATH,
    strategy      = "class_weight",
    augment_level = "full",
)

eff_model = build_efficientnet_model(pretrained=True)

eff_trainer = Trainer(
    model            = eff_model,
    train_loader     = train_loader2,
    val_loader       = val_loader2,
    class_weights    = class_weights2,
    model_name       = "efficientnet_b4",
    use_class_weight = True,
)

eff_history = eff_trainer.train(epochs=cfg.EPOCHS)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Ablation — Swin with no augmentation (baseline comparison)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXPERIMENT 3 — Swin-Base + NO augmentation (ablation)")
print("="*60)

train_loader3, val_loader3, _, _ = build_dataloaders(
    image_dir     = cfg.DATA_DIR + "/train",
    csv_path      = cfg.CSV_PATH,
    strategy      = "weighted_sampler",
    augment_level = "none",
)

swin_noaug = build_swin_model(pretrained=True)
trainer3   = Trainer(swin_noaug, train_loader3, val_loader3,
                     model_name="swin_noaug", use_class_weight=False)
trainer3.train(epochs=15)   # shorter ablation run


# ─────────────────────────────────────────────────────────────────────────────
# FINAL EVALUATION on test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(model, test_loader, model_name):
    """Load best checkpoint and evaluate on held-out test set."""
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{model_name}_best.pth")
    ckpt      = torch.load(ckpt_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nLoaded best {model_name} from epoch {ckpt['epoch']}")

    tracker = MetricTracker()
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            logits = model(images)
            tracker.update(logits, labels)

    metrics = tracker.compute()
    print_metrics(metrics, model_name)
    plot_confusion_matrix(
        metrics["labels"], metrics["preds"],
        model_name = model_name,
        save_path  = f"./logs/{model_name}_confusion.png"
    )
    return metrics


print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

swin_test_metrics = evaluate_on_test(swin_model, test_loader, "swin_base")
eff_test_metrics  = evaluate_on_test(eff_model, test_loader2, "efficientnet_b4")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE (paste into report)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("MODEL COMPARISON (test set)")
print("="*60)
print(f"{'Metric':<20} {'Swin-Base':>12} {'EfficientNet-B4':>16}")
print("-"*50)
for key in ["macro_f1", "macro_precision", "macro_recall", "kappa", "roc_auc"]:
    print(f"{key:<20} {swin_test_metrics[key]:>12.4f} {eff_test_metrics[key]:>16.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

from dataset import DRDataset, get_val_transforms
import pandas as pd

df       = pd.read_csv(cfg.CSV_PATH)
test_ds  = DRDataset(df.tail(50), cfg.DATA_DIR + "/train",
                     transform=get_val_transforms(cfg.IMAGE_SIZE))

# Target layer for Swin: last block of last stage
target_layer = swin_model.backbone.layers[-1].blocks[-1].norm1

visualise_gradcam_batch(
    model        = swin_model,
    dataset      = test_ds,
    indices      = [0, 10, 20, 30, 40],  # pick one of each DR grade if possible
    target_layer = target_layer,
    model_name   = "Swin-Base",
    save_path    = "./logs/swin_gradcam.png"
)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE FINAL EXPERIMENT LOG (upload to GitHub)
# ─────────────────────────────────────────────────────────────────────────────

experiment_log = {
    "swin_base": {
        "history"     : swin_history,
        "test_metrics": {k: v for k, v in swin_test_metrics.items()
                         if k not in ["preds", "labels", "probs"]},
        "strategy"    : "weighted_sampler",
        "augment"     : "full",
    },
    "efficientnet_b4": {
        "history"     : eff_history,
        "test_metrics": {k: v for k, v in eff_test_metrics.items()
                         if k not in ["preds", "labels", "probs"]},
        "strategy"    : "class_weight",
        "augment"     : "full",
    },
}

with open("./logs/experiment_log.json", "w") as f:
    json.dump(experiment_log, f, indent=2)
print("\nExperiment log saved → ./logs/experiment_log.json")
print("Upload this to your GitHub repo with a descriptive commit message.")
