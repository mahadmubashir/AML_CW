"""
EEEM068 — DR Grading
metrics.py — Evaluation metrics (accuracy explicitly excluded per feedback).

METRICS USED (as required by module leader):
  - Macro Precision, Recall, F1-score
  - Per-class F1
  - Cohen's Kappa (standard for ordinal medical grading)
  - ROC-AUC (one-vs-rest, macro)
  - Confusion matrix
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_preds  = []
        self.all_labels = []
        self.all_probs  = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        probs  = F.softmax(logits.detach().cpu(), dim=1).numpy()
        preds  = logits.argmax(dim=1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        self.all_probs.append(probs)
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def compute(self) -> dict:
        preds  = np.concatenate(self.all_preds)
        labels = np.concatenate(self.all_labels)
        probs  = np.concatenate(self.all_probs, axis=0)

        macro_f1  = f1_score(labels, preds, average='macro', zero_division=0)
        macro_pre = precision_score(labels, preds, average='macro', zero_division=0)
        macro_rec = recall_score(labels, preds, average='macro', zero_division=0)
        kappa     = cohen_kappa_score(labels, preds, weights='quadratic')
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

        try:
            auc = roc_auc_score(
                np.eye(5)[labels], probs,
                multi_class='ovr', average='macro'
            )
        except ValueError:
            auc = float('nan')

        return {
            "macro_f1"     : round(float(macro_f1), 4),
            "macro_precision": round(float(macro_pre), 4),
            "macro_recall" : round(float(macro_rec), 4),
            "kappa"        : round(float(kappa), 4),
            "roc_auc"      : round(float(auc), 4),
            "per_class_f1" : [round(float(f), 4) for f in per_class_f1],
            "preds"        : preds,
            "labels"       : labels,
            "probs"        : probs,
        }


def print_metrics(metrics: dict, model_name: str = "Model"):
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {metrics['macro_recall']:.4f}")
    print(f"  Cohen's Kappa  : {metrics['kappa']:.4f}  (quadratic weighted)")
    print(f"  ROC-AUC (OvR)  : {metrics['roc_auc']:.4f}")
    print(f"\n  Per-class F1:")
    grades = ["0-No DR", "1-Mild", "2-Moderate", "3-Severe", "4-Proliferative"]
    for cls, f1 in zip(grades, metrics['per_class_f1']):
        bar = "█" * int(f1 * 20)
        print(f"    {cls:18s}: {f1:.4f}  {bar}")
    print(f"{'='*50}\n")


def plot_confusion_matrix(labels, preds, model_name: str = "Model",
                           save_path: str = None):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    grades = ["No DR", "Mild", "Moderate", "Severe", "Prolif."]

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        [f"{model_name} — Confusion Matrix (counts)",
         f"{model_name} — Confusion Matrix (normalised)"],
        ["d", ".2f"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=grades, yticklabels=grades,
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_training_curves(train_losses, val_losses, val_f1s,
                          model_name: str = "Model", save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train loss", color="#2563EB")
    axes[0].plot(val_losses,   label="Val loss",   color="#DC2626")
    axes[0].set_title(f"{model_name} — Loss curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_f1s, label="Val macro F1", color="#16A34A")
    axes[1].set_title(f"{model_name} — Validation F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig
