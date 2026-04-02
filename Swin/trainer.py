"""
EEEM068 — DR Grading
trainer.py — Training loop with full experiment logging.

COVERS:
  - AdamW + cosine annealing LR schedule
  - Label smoothing loss
  - Gradient clipping
  - Class-weighted loss (imbalance strategy 1)
  - Best-model checkpointing on val macro F1
  - Weekly-ready training logs (upload to GitHub)
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from metrics import MetricTracker, print_metrics, plot_training_curves
from config import cfg


class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 class_weights=None, model_name: str = "model",
                 use_class_weight: bool = True):

        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.model_name   = model_name
        self.device       = cfg.DEVICE

        # Loss — with or without class weighting
        if use_class_weight and class_weights is not None:
            weight = class_weights.to(self.device)
        else:
            weight = None
        self.criterion = nn.CrossEntropyLoss(
            weight         = weight,
            label_smoothing= cfg.LABEL_SMOOTHING,
        )

        # Optimizer: layer-wise LR if model supports it
        if hasattr(model, 'get_param_groups'):
            param_groups = model.get_param_groups(cfg.LR)
        else:
            param_groups = model.parameters()

        self.optimizer = AdamW(
            param_groups,
            lr           = cfg.LR,
            weight_decay = cfg.WEIGHT_DECAY,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS, eta_min=1e-7)

        self.scaler = GradScaler()  # mixed precision

        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)

        self.history = {
            "train_loss": [], "val_loss": [],
            "val_f1": [], "val_kappa": [], "val_auc": [],
            "lr": []
        }
        self.best_f1   = 0.0
        self.best_epoch = 0

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.GRAD_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            n          += images.size(0)

        return total_loss / n

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        n          = 0
        tracker    = MetricTracker()

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast():
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            tracker.update(logits, labels)
            total_loss += loss.item() * images.size(0)
            n          += images.size(0)

        metrics = tracker.compute()
        return total_loss / n, metrics

    def _save_checkpoint(self, epoch: int, metrics: dict):
        path = os.path.join(cfg.CHECKPOINT_DIR,
                            f"{self.model_name}_best.pth")
        torch.save({
            "epoch"       : epoch,
            "model_state" : self.model.state_dict(),
            "optim_state" : self.optimizer.state_dict(),
            "metrics"     : metrics,
            "config"      : vars(cfg),
        }, path)
        print(f"  ✓ Checkpoint saved: {path}")

    def _save_log(self):
        path = os.path.join(cfg.LOG_DIR, f"{self.model_name}_log.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def train(self, epochs: int = None):
        epochs = epochs or cfg.EPOCHS
        print(f"\nTraining {self.model_name} for {epochs} epochs on {self.device}")
        print(f"Strategy: {cfg.IMBALANCE_STRATEGY} | Augment: full | LR: {cfg.LR}")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Warm-up: freeze backbone for first WARMUP_EPOCHS
            if epoch == 1 and hasattr(self.model, 'freeze_backbone'):
                self.model.freeze_backbone()
                print("  [Warm-up] Backbone frozen")
            if epoch == cfg.WARMUP_EPOCHS + 1 and hasattr(self.model, 'unfreeze_backbone'):
                self.model.unfreeze_backbone()
                print("  [Warm-up done] Backbone unfrozen — full fine-tuning")

            train_loss           = self._train_epoch()
            val_loss, metrics    = self._val_epoch()
            self.scheduler.step()

            cur_lr = self.optimizer.param_groups[-1]['lr']
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(metrics["macro_f1"])
            self.history["val_kappa"].append(metrics["kappa"])
            self.history["val_auc"].append(metrics["roc_auc"])
            self.history["lr"].append(cur_lr)

            improved = metrics["macro_f1"] > self.best_f1
            if improved:
                self.best_f1    = metrics["macro_f1"]
                self.best_epoch = epoch
                self._save_checkpoint(epoch, metrics)

            mark = "★" if improved else " "
            print(f"Ep {epoch:3d}/{epochs} {mark}| "
                  f"Loss {train_loss:.4f}/{val_loss:.4f} | "
                  f"F1 {metrics['macro_f1']:.4f} | "
                  f"Kappa {metrics['kappa']:.4f} | "
                  f"AUC {metrics['roc_auc']:.4f} | "
                  f"LR {cur_lr:.2e} | {elapsed:.0f}s")

        self._save_log()
        print(f"\nTraining complete. Best val F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
        plot_training_curves(
            self.history["train_loss"],
            self.history["val_loss"],
            self.history["val_f1"],
            model_name = self.model_name,
            save_path  = os.path.join(cfg.LOG_DIR, f"{self.model_name}_curves.png"),
        )
        return self.history
