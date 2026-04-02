"""
EEEM068 — DR Grading
gradcam.py — Grad-CAM visualisations (required for report + extra credit).

WHY THIS MATTERS FOR MARKS:
  - Satisfies "confusion matrices and model visualisations" requirement
  - Demonstrates the model is attending to clinically relevant regions
    (optic disc, macula, vessel abnormalities) — not spurious artefacts
  - Earns extra credit (interface/visualisation enhancements)
  - Viva: examiners WILL ask "how do you know your model isn't cheating?"
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import cfg


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Works for both Swin Transformer and EfficientNet:
      - Swin: target the last Swin stage's norm layer
      - EfficientNet: target the last convolutional block

    Usage:
        cam = GradCAM(model, target_layer=model.backbone.layers[-1].blocks[-1].norm1)
        heatmap = cam(image_tensor, class_idx=2)   # class 2 = Moderate DR
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        x = x.unsqueeze(0).to(cfg.DEVICE)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads  = self.gradients    # (1, N, C) for Swin, (1, C, H, W) for CNN
        acts   = self.activations  # same shape

        # Handle Swin output shape: (B, seq_len, channels) → need spatial reshape
        if grads.ndim == 3:
            # Swin: reshape from sequence to spatial grid
            B, N, C   = grads.shape
            grid_size = int(N ** 0.5)
            grads = grads.reshape(B, grid_size, grid_size, C).permute(0, 3, 1, 2)
            acts  = acts.reshape(B, grid_size, grid_size, C).permute(0, 3, 1, 2)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam     = (weights * acts).sum(dim=1).squeeze(0)
        cam     = F.relu(cam)
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam     = cv2.resize(cam, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        return cam, class_idx


def overlay_gradcam(original_img: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """Overlay Grad-CAM heatmap on the original fundus image."""
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * original_img).astype(np.uint8)
    return overlay


def visualise_gradcam_batch(model, dataset, indices: list,
                             target_layer, model_name: str = "Model",
                             save_path: str = None):
    """
    Visualise Grad-CAM for a batch of images.
    indices: list of dataset indices to visualise.
    """
    grades   = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    cam_obj  = GradCAM(model, target_layer)
    n        = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None]

    for row, idx in enumerate(indices):
        img_tensor, true_label = dataset[idx]
        cam, pred_label = cam_obj(img_tensor)

        # Recover original image for display
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        orig = img_tensor.permute(1, 2, 0).numpy()
        orig = (orig * std + mean).clip(0, 1)
        orig_uint8 = (orig * 255).astype(np.uint8)
        overlay    = overlay_gradcam(orig_uint8, cam)

        axes[row, 0].imshow(orig)
        axes[row, 0].set_title(f"Original | True: {grades[true_label]}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cam, cmap="jet")
        axes[row, 1].set_title(f"Grad-CAM | Pred: {grades[pred_label]}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"{model_name} — Overlay")
        axes[row, 2].axis("off")

    plt.suptitle(f"{model_name} — Grad-CAM Visualisations", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig
