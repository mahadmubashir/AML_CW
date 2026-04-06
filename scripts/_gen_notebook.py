"""One-off generator for notebooks/dr_training.ipynb.

Run:
  python scripts/_gen_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "dr_training.ipynb"

cells: list[dict] = []


def md(source: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.splitlines()],
        }
    )


def code(source: str) -> None:
    lines = source.splitlines()
    src = [line + "\n" for line in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src,
        }
    )


md(
    """# Diabetic retinopathy grading (multiclass)

**Task:** Predict one severity label per fundus image (0–4). This is **multiclass** classification.

**Workflow:** Edit the **Config** code cell below (paths, model, preprocessing, sampling, loss, hyperparameters), then run all cells top-to-bottom."""
)

code(
    """# 1) Environment and imports
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

# ImageNet normalization (matches torchvision pretrained weights)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
"""
)

md(
    """## 2) Configuration (single code cell)

This is the **only block** each teammate should edit. Everything else reads from these variables."""
)

code(
    """# 2) Config — edit only this cell per experiment.

# --- Paths (use external drive absolute paths if needed) ---
EXPERIMENT_NAME = "efficientnet_b0_baseline"
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
ARCHITECTURE = "efficientnet_b0"  # or vit_b_16, efficientnet_b4, ...
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
"""
)

code(
    """# Helpers shared across cells (seed, device, repo root)

def get_repo_root() -> Path:
    \"\"\"Resolve repo root whether the notebook cwd is repo root or notebooks/.\"\"\"\n
    cwd = Path.cwd().resolve()
    if (cwd / "data").is_dir() and (cwd / "notebooks").is_dir():
        return cwd
    if (cwd.parent / "data").is_dir() and (cwd.parent / "notebooks").is_dir():
        return cwd.parent
    return cwd


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
"""
)

md(
    """## 3) Dataset, preprocessing, transforms

`RetinaDataset` reads image paths from the label CSV. If `USE_PREPROCESSING=True`, it applies a fundus-specific preprocessing pipeline before standard transforms."""
)

code(
    """# 3) Dataset + preprocessing + transforms


def preprocess_fundus_pil(image: Image.Image, steps: dict) -> Image.Image:
    \"\"\"Best-effort approximation of the reference repo's preprocessing stages.\"\"\"\n
    img = np.array(image)  # RGB uint8
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if steps.get("crop_black", True):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img_bgr = img_bgr[y : y + h, x : x + w]

    if steps.get("circular_mask", True):
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        r = int(0.48 * min(h, w))
        cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    if steps.get("glare_inpaint", True):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        glare = (v > 245).astype(np.uint8) * 255
        if glare.sum() > 0:
            glare = cv2.dilate(glare, np.ones((3, 3), np.uint8), iterations=1)
            img_bgr = cv2.inpaint(img_bgr, glare, 3, cv2.INPAINT_TELEA)

    if steps.get("ben_graham", True):
        blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=10)
        img_bgr = cv2.addWeighted(img_bgr, 4, blur, -4, 128)

    if steps.get("clahe", True):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        img_bgr = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    if steps.get("green_enhance", True):
        b, g, r = cv2.split(img_bgr)
        g = cv2.equalizeHist(g)
        img_bgr = cv2.merge((b, g, r))

    if steps.get("vessel_frangi", False):
        green = cv2.split(img_bgr)[1].astype(np.float32) / 255.0
        vessels = frangi(green)
        vessels = (255 * (vessels / (vessels.max() + 1e-6))).astype(np.uint8)
        vessels_bgr = cv2.cvtColor(vessels, cv2.COLOR_GRAY2BGR)
        img_bgr = cv2.addWeighted(img_bgr, 0.8, vessels_bgr, 0.2, 0)

    if steps.get("red_enhance", False):
        b, g, r = cv2.split(img_bgr)
        r = cv2.equalizeHist(r)
        img_bgr = cv2.merge((b, g, r))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if steps.get("adaptive_denoise", True):
        img_dn = denoise_bilateral(img_rgb, sigma_color=0.05, sigma_spatial=3, channel_axis=-1)
        img_rgb = (np.clip(img_dn, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(img_rgb)


class RetinaDataset(Dataset):
    \"\"\"Fundus images + integer labels 0..NUM_CLASSES-1.\"\"\"\n
    def __init__(self, frame: pd.DataFrame, data_root: Path, image_extension: str, transform, preprocess: bool):
        self.frame = frame.reset_index(drop=True)
        self.data_root = Path(data_root)
        ext = image_extension if str(image_extension).startswith(".") else f".{image_extension}"
        self.image_extension = ext
        self.transform = transform
        self.preprocess = bool(preprocess)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        stem = str(row["image"])
        if not stem.lower().endswith((".jpeg", ".jpg", ".png")):
            stem = stem + self.image_extension
        path = self.data_root / stem
        image = Image.open(path).convert("RGB")
        if self.preprocess:
            image = preprocess_fundus_pil(image, PREPROCESS)
        label = int(row["level"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


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
"""
)

md("""## 4) Train / validation split""")
code(
    """# 4) Load labels CSV and stratified train/val split

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
print("Train label counts:\\n", train_df["level"].value_counts().sort_index())
"""
)

md("""## 5) DataLoaders (optional weighted sampling)""")
code(
    """# 5) Wrap datasets with DataLoader — optional weighted sampling

train_tf = build_transforms(train=True)
val_tf = build_transforms(train=False)

train_ds = RetinaDataset(train_df, data_root, IMAGE_EXTENSION, train_tf, preprocess=USE_PREPROCESSING)
val_ds = RetinaDataset(val_df, data_root, IMAGE_EXTENSION, val_tf, preprocess=USE_PREPROCESSING)

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
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=int(BATCH_SIZE),
    shuffle=False,
    num_workers=int(NUM_WORKERS),
    pin_memory=torch.cuda.is_available(),
)
"""
)

md("""## 6) Model architecture (EfficientNet / ViT)""")
code(
    """# 6) Build model — torchvision EfficientNet / ViT, NUM_CLASSES head


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
    else:
        for p in model.parameters():
            p.requires_grad = True


model = build_model().to(device)
"""
)

md("""## 7) Loss, optimizer, scheduler (label smoothing + focal loss)""")
code(
    """# 7) Loss + optimizer + scheduler


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
"""
)

md("""## 8) Training and validation""")
code(
    """# 8) Training loop — AMP optional, QWK, best checkpoint, early stopping

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
"""
)

md("""## 9) Evaluation on the validation split""")
code(
    """# 9) Load best weights and report confusion matrix + classification report

checkpoint = torch.load(best_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()

_, _, _, y_true, y_pred = run_epoch(val_loader, train_mode=False)
labels_names = [str(i) for i in range(int(NUM_CLASSES))]

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows=true, cols=pred):\\n", cm)
print("\\nClassification report:\\n", classification_report(y_true, y_pred, target_names=labels_names))

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
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", NB_PATH)

