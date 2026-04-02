"""
EEEM068 — DR Grading
swin_model.py — Swin Transformer implementation (YOUR model).

ARCHITECTURE JUSTIFICATION (for report + viva):
  Swin Transformer uses a hierarchical structure with shifted window
  attention. This gives it two key advantages over ViT for medical imaging:
  1. Local attention within windows preserves fine-grained lesion detail
     (microaneurysms, exudates) that global attention dilutes.
  2. Hierarchical feature maps (like CNNs) work well with multi-scale
     retinal features — optic disc (large) vs microaneurysm (tiny).
  3. Linear complexity O(N) vs ViT's O(N²) — efficient at 384px resolution.

WEAKNESSES TO DISCUSS:
  - Requires more GPU memory than EfficientNet at same resolution
  - Pre-training crucial — training from scratch on medical data alone is poor
  - Window size is a fixed inductive bias — may miss very long-range patterns
"""

import torch
import torch.nn as nn
import timm
from config import cfg


class SwinDRClassifier(nn.Module):
    """
    Swin Transformer B (swin_base_patch4_window12_384_in22k) fine-tuned
    for 5-class DR grading.

    Pre-trained on ImageNet-22k at 384×384 — directly compatible with
    retinal fundus images after Ben Graham preprocessing.

    Head design:
      Global average pooling → Dropout(0.3) → FC(5)
      Label smoothing applied during training via loss function.
    """

    def __init__(self, model_name: str = "swin_base_patch4_window12_384_in22k",
                 num_classes: int = 5, pretrained: bool = True,
                 drop_rate: float = 0.3):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained  = pretrained,
            num_classes = 0,          # remove timm's head — we add our own
            global_pool = "avg",
        )

        feature_dim = self.backbone.num_features  # 1024 for Swin-B

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, num_classes),
        )

        # Optionally freeze early stages for first few epochs
        self._frozen = False

    def freeze_backbone(self):
        """Freeze all backbone params — useful for warm-up phase."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze_backbone(self):
        """Unfreeze for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, 1024)
        logits   = self.head(features)  # (B, 5)
        return logits

    def get_param_groups(self, base_lr: float = 1e-4):
        """
        Layer-wise learning rate decay — backbone gets lower LR than head.
        Standard practice for fine-tuning transformers; prevents catastrophic
        forgetting of pre-trained representations.
        """
        return [
            {"params": self.backbone.parameters(), "lr": base_lr * 0.1},
            {"params": self.head.parameters(),     "lr": base_lr},
        ]


def build_swin_model(pretrained: bool = True) -> SwinDRClassifier:
    model = SwinDRClassifier(
        model_name  = "swin_base_patch4_window12_384_in22k",
        num_classes = cfg.NUM_CLASSES,
        pretrained  = pretrained,
    )
    model = model.to(cfg.DEVICE)
    total  = sum(p.numel() for p in model.parameters()) / 1e6
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Swin-Base | Total params: {total:.1f}M | Trainable: {train:.1f}M")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet-B4 Baseline — also in this file for easy comparison
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetDRClassifier(nn.Module):
    """
    EfficientNet-B4 baseline.

    Used as the shared baseline across all group members.
    Input size 380×380 (native B4 resolution).

    JUSTIFICATION:
      EfficientNet scales width, depth, and resolution uniformly via compound
      coefficient. B4 offers the best accuracy/efficiency trade-off for
      medical image tasks. Widely used as a strong CNN baseline in DR literature
      (Gulshan et al. 2016 used a similar compound-scaled architecture).

    WEAKNESSES:
      - Pure CNN — no explicit global attention like Swin's windows
      - Less effective at capturing global retinal structure
      - Fixed receptive field limits detection of very fine microaneurysms
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True,
                 drop_rate: float = 0.4):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained  = pretrained,
            num_classes = 0,
            global_pool = "avg",
        )

        feature_dim = self.backbone.num_features  # 1792 for B4

        self.head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_param_groups(self, base_lr: float = 1e-4):
        return [
            {"params": self.backbone.parameters(), "lr": base_lr * 0.1},
            {"params": self.head.parameters(),     "lr": base_lr},
        ]


def build_efficientnet_model(pretrained: bool = True) -> EfficientNetDRClassifier:
    model = EfficientNetDRClassifier(
        num_classes = cfg.NUM_CLASSES,
        pretrained  = pretrained,
    )
    model = model.to(cfg.DEVICE)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"EfficientNet-B4 | Total params: {total:.1f}M")
    return model
