"""One-off generator for dr_training.ipynb (run from repo root if needed)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "dr_training.ipynb"

cells = []

def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.splitlines()]})

def code(source: str):
    lines = source.splitlines()
    src = [line + "\n" for line in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

md("""# Diabetic retinopathy grading (multiclass)

**Task:** Predict severity label per fundus image (one label in $\\{0,1,2,3,4\\}$) — *multiclass* classification; cross-entropy is appropriate.

**Workflow:** Set `CONFIG_PATH` below to your YAML in `configs/`, then run all cells. Hyperparameters and architecture live in the config file.""")

code("""# 1) Environment and imports — libraries, device helpers, reproducibility utilities.
from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from contextlib import nullcontext
from PIL import Image
import yaml
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import torchvision.models as tv_models

# ImageNet normalization (matches torchvision pretrained weights)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
""")

md("""## 2) Configuration

Point `CONFIG_PATH` at **your** YAML file (paths are relative to the **repository root**). The loader deep-merges `extends` (e.g. `configs/base.yaml`). Change only this line between teammates.""")

code("""# 2) Load YAML config — single place to swap experiments.
CONFIG_PATH = "configs/efficientnet_b0_baseline.yaml"  # e.g. configs/member_1.yaml


def get_repo_root() -> Path:
    \"\"\"Resolve repo root whether the notebook cwd is repo root or notebooks/.\"\"\"
    cwd = Path.cwd().resolve()
    if (cwd / "configs").is_dir():
        return cwd
    if (cwd.parent / "configs").is_dir():
        return cwd.parent
    raise RuntimeError("Run from AML_G14_CW or AML_G14_CW/notebooks so configs/ is discoverable.")


REPO_ROOT = get_repo_root()
os.chdir(REPO_ROOT)
print("Repository root:", REPO_ROOT)


def deep_merge(base: dict, override: dict) -> dict:
    \"\"\"Recursively merge override into base (override wins). Skip 'extends' key.\"\"\"
    for key, value in override.items():
        if key == "extends":
            continue
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_rel_path: str, repo_root: Path) -> dict:
    \"\"\"Load YAML; if 'extends' is set, load base first and merge (child overrides).\"\"\"
    path = (repo_root / config_rel_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    extends = cfg.get("extends")
    if extends:
        base_cfg = load_config(str(extends), repo_root)
        merged = deep_merge(copy.deepcopy(base_cfg), cfg)
        merged.pop("extends", None)
        return merged
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


config = load_config(CONFIG_PATH, REPO_ROOT)
set_seed(config["reproducibility"]["seed"])
print("Experiment:", config.get("experiment_name"))
print("Model:", config["model"]["architecture"])
""")

md("""## 3) Dataset and transforms

`RetinaDataset` reads image paths from the label CSV. Training transforms include geometric and photometric augmentation controlled by `augmentation` in YAML; validation uses resize + center crop + normalize.""")

code("""# 3) PyTorch Dataset + transforms — resize/crop, aug on train only, RGB fundus images.


class RetinaDataset(Dataset):
    \"\"\"Fundus JPEGs + integer labels 0..num_classes-1.\"\"\"

    def __init__(self, frame: pd.DataFrame, data_root: Path, image_extension: str, transform):
        self.frame = frame.reset_index(drop=True)
        self.data_root = Path(data_root)
        ext = image_extension if str(image_extension).startswith(".") else f".{image_extension}"
        self.image_extension = ext
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        stem = str(row["image"])
        if not stem.lower().endswith((".jpeg", ".jpg", ".png")):
            stem = stem + self.image_extension
        path = self.data_root / stem
        image = Image.open(path).convert("RGB")
        label = int(row["level"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(cfg: dict, train: bool) -> transforms.Compose:
    size = int(cfg["data"]["image_size"])
    aug = cfg["augmentation"]
    if train:
        ops = [
            transforms.Resize(int(size * 1.12)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=float(aug["horizontal_flip_prob"])),
            transforms.RandomVerticalFlip(p=float(aug.get("vertical_flip_prob", 0.0))),
            transforms.RandomRotation(degrees=float(aug["rotation_degrees"])),
            transforms.ColorJitter(
                brightness=float(aug["color_jitter_brightness"]),
                contrast=float(aug["color_jitter_contrast"]),
                saturation=float(aug["color_jitter_saturation"]),
                hue=float(aug["color_jitter_hue"]),
            ),
            transforms.ToTensor(),
        ]
        if float(aug.get("random_erase_prob", 0.0)) > 0:
            ops.append(transforms.RandomErasing(p=float(aug["random_erase_prob"])))
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
""")

md("""## 4) Train / validation split

**Stratified** split keeps class proportions stable (important for imbalanced DR data). *Note:* Kaggle image IDs pair left/right eyes per patient; ideal leakage control is **patient-level** grouping — this skeleton uses image-level stratified split for simplicity; document that limitation if you adopt it. CSV columns are normalized to `image` and `level`.""")

code("""# 4) Load labels CSV and stratified train/val split.

paths_cfg = config["paths"]
labels_path = (REPO_ROOT / paths_cfg["labels_csv"]).resolve()
data_root = (REPO_ROOT / paths_cfg["data_root"]).resolve()

full_df = pd.read_csv(labels_path)
full_df.columns = [c.strip().lower() for c in full_df.columns]
assert "image" in full_df.columns and "level" in full_df.columns, (
    f"Expected columns image, level; got {full_df.columns.tolist()}"
)

train_df, val_df = train_test_split(
    full_df,
    test_size=float(config["reproducibility"]["train_val_split"]),
    stratify=full_df["level"],
    random_state=int(config["reproducibility"]["seed"]),
)
print("Train samples:", len(train_df), "Val samples:", len(val_df))
print("Train label counts:\\n", train_df["level"].value_counts().sort_index())
""")

md("""## 5) DataLoaders

Batching and shuffling for training; no shuffle on validation. `num_workers` is configurable for faster loading on Linux; use `0` on some macOS setups if you see multiprocessing issues.""")

code("""# 5) Wrap datasets with DataLoader — batching, shuffle (train only).

train_tf = build_transforms(config, train=True)
val_tf = build_transforms(config, train=False)

train_ds = RetinaDataset(
    train_df,
    data_root,
    paths_cfg["image_extension"],
    train_tf,
)
val_ds = RetinaDataset(
    val_df,
    data_root,
    paths_cfg["image_extension"],
    val_tf,
)

num_workers = int(config["reproducibility"]["num_workers"])

train_loader = DataLoader(
    train_ds,
    batch_size=int(config["training"]["batch_size"]),
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=int(config["training"]["batch_size"]),
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
""")

md("""## 6) Model architecture

`build_model` selects **EfficientNet** or **ViT** from `torchvision.models` based on `model.architecture` in YAML; the classifier head is replaced for 5 classes. Dropout on EfficientNet follows torchvision; ViT gets a fresh linear head (optional dropout via `model.dropout` applied only on EfficientNet tail if you extend the code).""")

code("""# 6) Build model from config — torchvision EfficientNet / ViT, 5-class head.


def replace_classifier_head(model: nn.Module, arch: str, num_classes: int, dropout: float) -> None:
    arch_lower = arch.lower()
    if arch_lower.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        if dropout and dropout > 0:
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch_lower.startswith("vit"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture for head replacement: {arch}")


def build_model(cfg: dict) -> nn.Module:
    arch = cfg["model"]["architecture"]
    pretrained = bool(cfg["model"]["pretrained"])
    num_classes = int(cfg["data"]["num_classes"])
    dropout = float(cfg["model"].get("dropout", 0.0))
    weights_arg = "DEFAULT" if pretrained else None
    if not hasattr(tv_models, arch):
        raise ValueError(
            f"Unknown torchvision model '{arch}'. Use e.g. efficientnet_b0 … vit_b_16 (see configs/README)."
        )
    ctor = getattr(tv_models, arch)
    model = ctor(weights=weights_arg)
    replace_classifier_head(model, arch, num_classes, dropout)
    return model


def set_backbone_trainable(model: nn.Module, architecture: str, train_backbone: bool) -> None:
    \"\"\"If train_backbone False, freeze feature encoder; always train classification head.\"\"\"
    arch = architecture.lower()
    if arch.startswith("vit"):
        for name, param in model.named_parameters():
            if name.startswith("heads"):
                param.requires_grad = True
            else:
                param.requires_grad = train_backbone
    elif arch.startswith("efficientnet"):
        for name, param in model.named_parameters():
            if name.startswith("features"):
                param.requires_grad = train_backbone
            else:
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True


model = build_model(config)
""")

md("""## 7) Loss, optimizer, scheduler

Cross-entropy with optional **class weights** (`loss.class_weights: auto` uses balanced weights from training labels). Optimizer and cosine / step scheduler are driven by YAML.""")

code("""# 7) Loss with optional class weights; AdamW/SGD; lr scheduler from config.


def resolve_device(spec: str) -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


device = resolve_device(config["reproducibility"].get("device", "auto"))
model = model.to(device)

freeze_epochs_pre = int(config["model"].get("freeze_backbone_epochs", 0))
set_backbone_trainable(
    model,
    config["model"]["architecture"],
    train_backbone=(freeze_epochs_pre == 0),
)

loss_cfg = config["loss"]
num_classes = int(config["data"]["num_classes"])
if loss_cfg.get("class_weights") == "auto":
    y_train = train_df["level"].to_numpy()
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (len(y_train) / (num_classes * counts)).astype(np.float32)
    class_weights = torch.tensor(weights, device=device)
elif loss_cfg.get("class_weights"):
    class_weights = torch.tensor(loss_cfg["class_weights"], dtype=torch.float32, device=device)
else:
    class_weights = None

if loss_cfg.get("type", "cross_entropy") != "cross_entropy":
    raise ValueError("Only cross_entropy is implemented in this skeleton.")

criterion = nn.CrossEntropyLoss(weight=class_weights)

tr_cfg = config["training"]
params = [p for p in model.parameters() if p.requires_grad]
if tr_cfg["optimizer"].lower() == "adamw":
    optimizer = torch.optim.AdamW(params, lr=float(tr_cfg["lr"]), weight_decay=float(tr_cfg["weight_decay"]))
elif tr_cfg["optimizer"].lower() == "sgd":
    optimizer = torch.optim.SGD(
        params,
        lr=float(tr_cfg["lr"]),
        momentum=0.9,
        weight_decay=float(tr_cfg["weight_decay"]),
    )
else:
    raise ValueError(f"Unknown optimizer: {tr_cfg['optimizer']}")

scheduler = None
sched_name = tr_cfg.get("scheduler", "none")
if sched_name == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(tr_cfg["epochs"])
    )
elif sched_name == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(tr_cfg.get("scheduler_step_size", 7)),
        gamma=float(tr_cfg.get("scheduler_gamma", 0.1)),
    )
""")

md("""## 8) Training and validation

Tracks loss, accuracy, and **quadratic weighted kappa** (competition-aligned). Mixed precision is optional (`training.mixed_precision`). Saves best checkpoint to `checkpoints/` by `save_best_metric` (`val_kappa` or `val_accuracy`). Early stopping uses `early_stopping_patience`. **Phase 1:** if `model.freeze_backbone_epochs` > 0, backbone is frozen for those epochs then unfrozen.""")

code("""# 8) Training loop — AMP optional, QWK, best checkpoint, early stopping.

tr_cfg = config["training"]
use_amp = bool(tr_cfg.get("mixed_precision", False)) and device.type == "cuda"
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
        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=True)
            if use_amp
            else nullcontext()
        )
        with autocast_ctx:
            logits = model(images)
            loss = criterion(logits, labels)
        if train_mode:
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if tr_cfg.get("gradient_clip_norm"):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), float(tr_cfg["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if tr_cfg.get("gradient_clip_norm"):
                    nn.utils.clip_grad_norm_(model.parameters(), float(tr_cfg["gradient_clip_norm"]))
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


ckpt_dir = (REPO_ROOT / paths_cfg["checkpoint_dir"]).resolve()
ckpt_dir.mkdir(parents=True, exist_ok=True)
best_path = ckpt_dir / f"{config.get('experiment_name', 'run')}_best.pt"

freeze_epochs = int(config["model"].get("freeze_backbone_epochs", 0))
best_metric_val = -1.0
patience_left = int(tr_cfg.get("early_stopping_patience", 9999))
metric_name = tr_cfg.get("save_best_metric", "val_kappa")
sched_name = tr_cfg.get("scheduler", "none")

for epoch in range(int(tr_cfg["epochs"])):
    train_backbone = epoch >= freeze_epochs
    set_backbone_trainable(
        model,
        config["model"]["architecture"],
        train_backbone=train_backbone,
    )
    if train_backbone and epoch == freeze_epochs and freeze_epochs > 0:
        params = [p for p in model.parameters() if p.requires_grad]
        opt_name = tr_cfg["optimizer"].lower()
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=float(tr_cfg["lr"]),
                weight_decay=float(tr_cfg["weight_decay"]),
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=float(tr_cfg["lr"]),
                momentum=0.9,
                weight_decay=float(tr_cfg["weight_decay"]),
            )
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(tr_cfg["epochs"]) - epoch),
            )
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(tr_cfg.get("scheduler_step_size", 7)),
                gamma=float(tr_cfg.get("scheduler_gamma", 0.1)),
            )

    train_loss, train_acc, train_kappa, _, _ = run_epoch(train_loader, train_mode=True)
    val_loss, val_acc, val_kappa, _, _ = run_epoch(val_loader, train_mode=False)
    if scheduler is not None:
        scheduler.step()

    print(
        f"Epoch {epoch+1}/{tr_cfg['epochs']} | "
        f"train loss {train_loss:.4f} acc {train_acc:.4f} kappa {train_kappa:.4f} | "
        f"val loss {val_loss:.4f} acc {val_acc:.4f} kappa {val_kappa:.4f}"
    )

    if metric_name == "val_kappa":
        tracked = val_kappa
    else:
        tracked = val_acc
    if np.isnan(tracked):
        tracked = -1.0

    if tracked > best_metric_val:
        best_metric_val = tracked
        patience_left = int(tr_cfg.get("early_stopping_patience", 9999))
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_kappa": val_kappa,
                "val_acc": val_acc,
            },
            best_path,
        )
        print(f"  Saved new best ({metric_name}={tracked:.4f}) -> {best_path}")
    else:
        patience_left -= 1
        if patience_left <= 0:
            print("Early stopping.")
            break

print("Training finished. Best checkpoint:", best_path)
""")

md("""## 9) Evaluation on the validation split

Confusion matrix and per-class precision/recall after loading the **best** weights.""")

code("""# 9) Load best weights and report confusion matrix + classification report.

checkpoint = torch.load(best_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()

_, _, _, y_true, y_pred = run_epoch(val_loader, train_mode=False)
labels_names = [str(i) for i in range(int(config["data"]["num_classes"]))]

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows=true, cols=pred):\\n", cm)
print("\\nClassification report:\\n", classification_report(y_true, y_pred, target_names=labels_names))

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation="nearest")
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xlabel="Predicted", ylabel="True", title="Confusion matrix")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
fig.tight_layout()
out_dir = (REPO_ROOT / paths_cfg["outputs_dir"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
fig_path = out_dir / f"{config.get('experiment_name', 'run')}_confusion.png"
fig.savefig(fig_path, dpi=150)
print("Saved figure to", fig_path)
plt.show()
""")

md("""## 10) Optional: Kaggle-style test inference

Runs when `paths.test_dir` and `paths.sample_submission_csv` are set in YAML; writes `outputs/submission.csv`. Leave both `null` in `configs/base.yaml` to skip (this cell will no-op if missing).""")

code("""# 10) Optional inference on test folder — fills submission CSV template if configured.

test_dir_cfg = paths_cfg.get("test_dir")
sample_csv = paths_cfg.get("sample_submission_csv")

if test_dir_cfg and sample_csv:
    test_root = (REPO_ROOT / test_dir_cfg).resolve()
    sub_df = pd.read_csv((REPO_ROOT / sample_csv).resolve())
    sub_df.columns = [c.strip().lower() for c in sub_df.columns]
    model.eval()
    test_tf = build_transforms(config, train=False)
    preds = []
    for stem in tqdm(sub_df["image"].astype(str)):
        name = stem if stem.lower().endswith((".jpeg", ".jpg", ".png")) else stem + paths_cfg["image_extension"]
        path = test_root / name
        img = Image.open(path).convert("RGB")
        x = test_tf(img).unsqueeze(0).to(device)
        preds.append(int(model(x).argmax(dim=1).item()))
    out_col = "level" if "level" in sub_df.columns else sub_df.columns[-1]
    sub_df[out_col] = preds
    sub_path = out_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print("Wrote", sub_path)
else:
    print("Skipping test inference (set paths.test_dir and paths.sample_submission_csv in config).")
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}

NB_path = ROOT / "notebooks"
NB_path.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", NB_PATH)
