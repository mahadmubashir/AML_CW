"""
EEEM068 — Diabetic Retinopathy Grading
Shared configuration — all group members use this file unchanged.
"""

import torch

class Config:
    # Dataset
    DATA_DIR        = "/content/drive/MyDrive/DR_dataset"   # update per member
    CSV_PATH        = f"{DATA_DIR}/trainLabels.csv"
    IMAGE_SIZE      = 384          # Swin-Base input; EfficientNet-B4 uses 380
    BATCH_SIZE      = 16
    NUM_WORKERS     = 4
    NUM_CLASSES     = 5            # DR grades 0-4

    # Training
    EPOCHS          = 30
    LR              = 1e-4
    WEIGHT_DECAY    = 1e-2
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP       = 1.0
    WARMUP_EPOCHS   = 3

    # Reproducibility
    SEED            = 42

    # Imbalance strategy — set per experiment
    # Options: "class_weight", "weighted_sampler", "oversample"
    IMBALANCE_STRATEGY = "weighted_sampler"

    # Paths
    CHECKPOINT_DIR  = "./checkpoints"
    LOG_DIR         = "./logs"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
