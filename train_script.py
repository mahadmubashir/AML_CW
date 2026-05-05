#!/usr/bin/env python
# coding: utf-8

# # Diabetic retinopathy grading (multiclass)
# 
# **Task:** Predict one severity label per fundus image (0–4). This is **multiclass** classification.
# 
# **Workflow:** Edit the **Config** code cell below (paths, model, preprocessing, sampling, loss, hyperparameters), then run all cells top-to-bottom.
# 

# In[ ]:


# 1) Environment and imports
#
# All experiment knobs live in the Config cell.
from __future__ import annotations

import os
import random
from contextlib import nullcontext
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.filters import frangi
from skimage.restoration import denoise_bilateral
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm.auto import tqdm

import torchvision.models as tv_models

# repo-aware path resolver — works whether cwd is repo root or notebooks/
import sys as _sys
from pathlib import Path as _Path
_cwd = _Path.cwd().resolve()
_repo = _cwd.parent if _cwd.name == "notebooks" else _cwd
_sys.path.insert(0, str(_repo))

# import dataset class from real .py module so spawn workers can pickle it
from src.retina_dataset import RetinaDataset, preprocess_fundus_pil

# ImageNet normalization (matches torchvision pretrained weights)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ## 2) Configuration (single code cell)
# 
# This is the **only block** each teammate should edit. Everything else reads from these variables.
# 

# In[ ]:


# 2) Config — edit only this cell per experiment.

# --- Paths (use external drive absolute paths if needed) ---
EXPERIMENT_NAME = "mobilenetv3_large_na02153"
DATA_ROOT = "data/train"             # folder containing images
LABELS_CSV = "data/trainLabels.csv"  # CSV with columns: image, level
IMAGE_EXTENSION = ".jpeg"

# --- Split / reproducibility ---
SEED = 42
TRAIN_VAL_SPLIT = 0.2
NUM_WORKERS = 4
DEVICE = "auto"  # "auto" | "cuda" | "mps" | "cpu"

# --- Data / augmentation ---
IMAGE_SIZE = 224
NUM_CLASSES = 5

HFLIP_PROB = 0.5
VFLIP_PROB = 0.0
ROTATION_DEG = 15
COLOR_JITTER = dict(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
RANDOM_ERASE_PROB = 0.0

# --- Fundus preprocessing (reference repo style; heavy) ---
USE_PREPROCESSING = False
PREPROCESS = dict(
    crop_black=True,
    circular_mask=True,
    glare_inpaint=True,
    ben_graham=True,
    clahe=True,
    green_enhance=True,
    vessel_frangi=False,  # expensive; off by default
    red_enhance=False,
    adaptive_denoise=True,
)

# --- Model ---
ARCHITECTURE = "mobilenet_v3_large"  # MobileNetV3-Large pretrained on ImageNet
PRETRAINED = True
DROPOUT = 0.2
FREEZE_BACKBONE_EPOCHS = 0

# --- Training ---
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
OPTIMIZER = "adamw"  # "adamw" | "sgd"
WEIGHT_DECAY = 1e-2
SCHEDULER = "cosine"  # "cosine" | "step" | "none"
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA = 0.1
EARLY_STOPPING_PATIENCE = 5
MIXED_PRECISION = True          # only used on CUDA
GRADIENT_CLIP_NORM = None
SAVE_BEST_METRIC = "val_kappa"  # "val_kappa" | "val_accuracy"

# --- Imbalance + loss toggles ---
CLASS_WEIGHTS_AUTO = True
WEIGHTED_SAMPLING = False

LABEL_SMOOTHING = 0.0  # 0.0 disables label smoothing
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None  # set e.g. 0.25 or keep None


# In[ ]:


# Helpers shared across cells (seed, device, repo root)

def get_repo_root() -> Path:
    """Resolve repo root whether the notebook cwd is repo root or notebooks/."""

    cwd = Path.cwd().resolve()
    if (cwd / "data").is_dir() and (cwd / "notebooks").is_dir():
        return cwd
    if (cwd.parent / "data").is_dir() and (cwd.parent / "notebooks").is_dir():
        return cwd.parent
    return cwd



def main():

    REPO_ROOT = get_repo_root()
    os.chdir(REPO_ROOT)
    print("Repository root:", REPO_ROOT)


    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def resolve_device(spec: str) -> torch.device:
        spec = (spec or "auto").lower()
        if spec == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(spec)


    set_seed(SEED)
    device = resolve_device(DEVICE)
    print("Device:", device)


    # ## 3) Dataset, preprocessing, transforms
    # 
    # `RetinaDataset` reads image paths from the label CSV. If `USE_PREPROCESSING=True`, it applies a fundus-specific preprocessing pipeline before standard transforms.
    # 

    # In[ ]:


    # 3) Dataset class is imported from src/retina_dataset.py at the top.
    # Only the transforms helper lives here because it depends on notebook-level
    # config constants (IMAGE_SIZE, HFLIP_PROB, COLOR_JITTER, etc.)


    def build_transforms(train: bool) -> transforms.Compose:
        size = int(IMAGE_SIZE)
        if train:
            ops = [
                transforms.Resize(int(size * 1.12)),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=float(HFLIP_PROB)),
                transforms.RandomVerticalFlip(p=float(VFLIP_PROB)),
                transforms.RandomRotation(degrees=float(ROTATION_DEG)),
                transforms.ColorJitter(
                    brightness=float(COLOR_JITTER["brightness"]),
                    contrast=float(COLOR_JITTER["contrast"]),
                    saturation=float(COLOR_JITTER["saturation"]),
                    hue=float(COLOR_JITTER["hue"]),
                ),
                transforms.ToTensor(),
            ]
            if float(RANDOM_ERASE_PROB) > 0:
                ops.append(transforms.RandomErasing(p=float(RANDOM_ERASE_PROB)))
            ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
            return transforms.Compose(ops)

        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )


    # ## 4) Train / validation split
    # 

    # In[ ]:


    # 4) Load labels CSV and stratified train/val split

    labels_path = (REPO_ROOT / LABELS_CSV).resolve() if not str(LABELS_CSV).startswith("/") else Path(LABELS_CSV)
    data_root = (REPO_ROOT / DATA_ROOT).resolve() if not str(DATA_ROOT).startswith("/") else Path(DATA_ROOT)

    full_df = pd.read_csv(labels_path)
    full_df.columns = [c.strip().lower() for c in full_df.columns]
    assert "image" in full_df.columns and "level" in full_df.columns, (
        f"Expected columns image, level; got {full_df.columns.tolist()}"
    )

    train_df, val_df = train_test_split(
        full_df,
        test_size=float(TRAIN_VAL_SPLIT),
        stratify=full_df["level"],
        random_state=int(SEED),
    )
    print("Train samples:", len(train_df), "Val samples:", len(val_df))
    print("Train label counts:\n", train_df["level"].value_counts().sort_index())


    # ## 5) DataLoaders (optional weighted sampling)
    # 

    # In[ ]:


    # 5) Wrap datasets with DataLoader — optional weighted sampling

    train_tf = build_transforms(train=True)
    val_tf = build_transforms(train=False)

    train_ds = RetinaDataset(train_df, data_root, IMAGE_EXTENSION, train_tf, preprocess=USE_PREPROCESSING, preprocess_steps=PREPROCESS)
    val_ds = RetinaDataset(val_df, data_root, IMAGE_EXTENSION, val_tf, preprocess=USE_PREPROCESSING, preprocess_steps=PREPROCESS)

    train_sampler = None
    if WEIGHTED_SAMPLING:
        counts = train_df["level"].value_counts().to_dict()
        weights = train_df["level"].map(lambda y: 1.0 / counts[int(y)]).to_numpy(dtype=np.float64)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(BATCH_SIZE),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(NUM_WORKERS),
        multiprocessing_context="spawn" if int(NUM_WORKERS) > 0 else None,
        persistent_workers=int(NUM_WORKERS) > 0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(BATCH_SIZE),
        shuffle=False,
        num_workers=int(NUM_WORKERS),
        multiprocessing_context="spawn" if int(NUM_WORKERS) > 0 else None,
        persistent_workers=int(NUM_WORKERS) > 0,
        pin_memory=False,
    )


    # ## 6) Model architecture (EfficientNet / ViT)
    # 

    # In[ ]:


    # 6) Build model — torchvision EfficientNet / ViT, NUM_CLASSES head


    def replace_classifier_head(model: nn.Module, arch: str, num_classes: int, dropout: float) -> None:
        arch_lower = arch.lower()
        if arch_lower.startswith("efficientnet"):
            in_features = model.classifier[1].in_features
            if dropout and dropout > 0:
                model.classifier = nn.Sequential(
                    nn.Dropout(p=float(dropout), inplace=True),
                    nn.Linear(in_features, num_classes),
                )
            else:
                model.classifier[1] = nn.Linear(in_features, num_classes)
        elif arch_lower.startswith("vit"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        elif arch_lower.startswith("mobilenet_v3"):
            # MobileNetV3 head: classifier[3] is the final Linear layer
            in_features = model.classifier[3].in_features
            if dropout and dropout > 0:
                model.classifier[2] = nn.Dropout(p=float(dropout), inplace=True)
            model.classifier[3] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")


    def build_model() -> nn.Module:
        if not hasattr(tv_models, ARCHITECTURE):
            raise ValueError(f"Unknown torchvision model '{ARCHITECTURE}'.")
        ctor = getattr(tv_models, ARCHITECTURE)
        weights_arg = "DEFAULT" if bool(PRETRAINED) else None
        m = ctor(weights=weights_arg)
        replace_classifier_head(m, ARCHITECTURE, int(NUM_CLASSES), float(DROPOUT))
        return m


    def set_backbone_trainable(model: nn.Module, architecture: str, train_backbone: bool) -> None:
        arch = architecture.lower()
        if arch.startswith("vit"):
            for name, p in model.named_parameters():
                if name.startswith("heads"):
                    p.requires_grad = True
                else:
                    p.requires_grad = bool(train_backbone)
        elif arch.startswith("efficientnet"):
            for name, p in model.named_parameters():
                if name.startswith("features"):
                    p.requires_grad = bool(train_backbone)
                else:
                    p.requires_grad = True
        elif arch.startswith("mobilenet_v3"):
            # MobileNetV3 layout: features = backbone, classifier = head
            for name, p in model.named_parameters():
                if name.startswith("features"):
                    p.requires_grad = bool(train_backbone)
                else:
                    p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True


    model = build_model().to(device)


    # ## 7) Loss, optimizer, scheduler (label smoothing + focal loss)
    # 

    # In[ ]:


    # 7) Loss + optimizer + scheduler


    def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
        counts = np.bincount(y, minlength=num_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = (len(y) / (num_classes * counts)).astype(np.float32)
        return torch.tensor(weights, device=device)


    class FocalLoss(nn.Module):
        def __init__(self, gamma: float = 2.0, alpha: float | None = None, weight: torch.Tensor | None = None):
            super().__init__()
            self.gamma = float(gamma)
            self.alpha = alpha if alpha is None else float(alpha)
            self.weight = weight

        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
            pt = torch.exp(-ce)
            loss = ((1 - pt) ** self.gamma) * ce
            if self.alpha is not None:
                loss = self.alpha * loss
            return loss.mean()


    set_backbone_trainable(model, ARCHITECTURE, train_backbone=(int(FREEZE_BACKBONE_EPOCHS) == 0))

    class_weights = None
    if CLASS_WEIGHTS_AUTO:
        class_weights = compute_class_weights(train_df["level"].to_numpy(), int(NUM_CLASSES))

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=float(FOCAL_GAMMA), alpha=FOCAL_ALPHA, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(LABEL_SMOOTHING))

    params = [p for p in model.parameters() if p.requires_grad]
    if OPTIMIZER.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, lr=float(LR), weight_decay=float(WEIGHT_DECAY))
    elif OPTIMIZER.lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=float(LR), momentum=0.9, weight_decay=float(WEIGHT_DECAY))
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    scheduler = None
    if SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(EPOCHS))
    elif SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(SCHEDULER_STEP_SIZE), gamma=float(SCHEDULER_GAMMA)
        )


    # ## 8) Training and validation
    # 

    # In[ ]:


    # 8) Training loop — AMP optional, QWK, best checkpoint, early stopping

    use_amp = bool(MIXED_PRECISION) and device.type == "cuda"
    train_state = {"scaler": torch.cuda.amp.GradScaler() if use_amp else None}


    def run_epoch(loader, train_mode: bool):
        model.train(train_mode)
        scaler = train_state["scaler"]
        total_loss = 0.0
        correct = 0
        total = 0
        preds_list = []
        labels_list = []
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            autocast_ctx = torch.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
            with autocast_ctx:
                logits = model(images)
                loss = criterion(logits, labels)

            if train_mode:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    if GRADIENT_CLIP_NORM:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), float(GRADIENT_CLIP_NORM))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if GRADIENT_CLIP_NORM:
                        nn.utils.clip_grad_norm_(model.parameters(), float(GRADIENT_CLIP_NORM))
                    optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += labels.size(0)
            preds_list.append(pred.detach().cpu())
            labels_list.append(labels.detach().cpu())

        y_pred = torch.cat(preds_list).numpy()
        y_true = torch.cat(labels_list).numpy()
        acc = correct / max(total, 1)
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        avg_loss = total_loss / max(total, 1)
        return avg_loss, acc, kappa, y_true, y_pred


    ckpt_dir = (REPO_ROOT / "checkpoints").resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{EXPERIMENT_NAME}_best.pt"

    freeze_epochs = int(FREEZE_BACKBONE_EPOCHS)
    best_metric_val = -1.0
    patience_left = int(EARLY_STOPPING_PATIENCE)
    metric_name = str(SAVE_BEST_METRIC)

    for epoch in range(int(EPOCHS)):
        train_backbone = epoch >= freeze_epochs
        set_backbone_trainable(model, ARCHITECTURE, train_backbone=train_backbone)

        if train_backbone and epoch == freeze_epochs and freeze_epochs > 0:
            params = [p for p in model.parameters() if p.requires_grad]
            if OPTIMIZER.lower() == "adamw":
                optimizer = torch.optim.AdamW(params, lr=float(LR), weight_decay=float(WEIGHT_DECAY))
            else:
                optimizer = torch.optim.SGD(params, lr=float(LR), momentum=0.9, weight_decay=float(WEIGHT_DECAY))
            if SCHEDULER == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(EPOCHS) - epoch))
            elif SCHEDULER == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=int(SCHEDULER_STEP_SIZE), gamma=float(SCHEDULER_GAMMA)
                )

        train_loss, train_acc, train_kappa, _, _ = run_epoch(train_loader, train_mode=True)
        val_loss, val_acc, val_kappa, _, _ = run_epoch(val_loader, train_mode=False)
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} kappa {train_kappa:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} kappa {val_kappa:.4f}"
        )

        tracked = val_kappa if metric_name == "val_kappa" else val_acc
        if np.isnan(tracked):
            tracked = -1.0

        if tracked > best_metric_val:
            best_metric_val = tracked
            patience_left = int(EARLY_STOPPING_PATIENCE)
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_kappa": val_kappa, "val_acc": val_acc},
                best_path,
            )
            print(f"  Saved new best ({metric_name}={tracked:.4f}) -> {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    print("Training finished. Best checkpoint:", best_path)


    # ## 9) Evaluation on the validation split
    # 

    # In[ ]:


    # 9) Load best weights and report confusion matrix + classification report

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, _, _, y_true, y_pred = run_epoch(val_loader, train_mode=False)
    labels_names = [str(i) for i in range(int(NUM_CLASSES))]

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, labels=list(range(int(NUM_CLASSES))), target_names=labels_names, zero_division=0))

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xlabel="Predicted",
        ylabel="True",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    out_dir = (REPO_ROOT / "outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{EXPERIMENT_NAME}_confusion.png"
    fig.savefig(fig_path, dpi=150)
    print("Saved figure to", fig_path)
    plt.show()



if __name__ == "__main__":
    main()
