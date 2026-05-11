# AML G14 — Diabetic Retinopathy Grading (EEEM068)

**University of Surrey — Applied Machine Learning Coursework**

5-class severity grading (Grade 0–4) of fundus retinal images using Vision Transformers.
Best model: **ViT-B/16** fine-tuned with two-phase training, FocalLoss(α=None, γ=2) + WeightedRandomSampler.

## Quick Start

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate          
pip install -r requirements.txt 

```

### 2. Dataset

Download from the [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) competition and extract into `data/diabetic-retinopathy-detection/`:

```
data/diabetic-retinopathy-detection/
├── train/                  ← raw training JPEGs
├── test/                   ← raw test JPEGs
├── test_labels/            ← test labels folder
├── trainLabels.csv         ← image, level columns
└── sampleSubmission.csv
```

### 3. Run notebooks in order

| Step | Notebook | Purpose |
|---|---|---|
| 1 | `notebooks/data_preprocessing.ipynb` | Ben Graham preprocessing → `train_proc/`, `val_proc/`, `test_labels_proc/` + CSV splits |
| 2 | `notebooks/baseline.ipynb` | EfficientNet-B0 baseline (Experiment 1) |
| 3 | `notebooks/vit_b16_initial_experiments.ipynb` | Experiments 1–8 (ablation study) |
| 4 | `notebooks/vit_b16_final.ipynb` | Best configuration + full explainability |

---

## Repository Layout

```
AML_G14_CW/
│
├── notebooks/
│   ├── data_preprocessing.ipynb          # data loading, EDA, Ben Graham preprocessing, train/val/test splits
│   ├── baseline.ipynb                     # EfficientNet-B0 baseline experiment
│   ├── vit_b16_initial_experiments.ipynb  # Experiments 1–8 ablation study
│   └── vit_b16_final.ipynb                # Final ViT-B/16 model + explainability (Sections 1–8)
│
├── data/
│   └── diabetic-retinopathy-detection/
│       ├── train/                          # raw training images (gitignored)
│       ├── train_proc/                     # preprocessed training images (gitignored)
│       ├── val/                            # raw validation images (gitignored)
│       ├── val_proc/                       # preprocessed validation images (gitignored)
│       ├── test_labels/                    # raw test images (gitignored)
│       ├── test_labels_proc/               # preprocessed test images (gitignored)
│       ├── train_split.csv                 # fixed train split (image, level)
│       ├── val_split.csv                   # fixed val split
│       └── test_split.csv                  # fixed test split
│
├── outputs/
│   ├── figures/                            # EDA and preprocessing figures
│   ├── baseline/                           # EfficientNet-B0 checkpoints + figures
│   ├── vit_b16/                            # scratch outputs from early runs
│   ├── vit_b16_exp1/ … vit_b16_exp8/      # per-experiment checkpoints, figures, CSV logs
│   └── vit_b16_final/
│       ├── checkpoints/
│       │   ├── vit_b16_final_phase1_best.pt   # Phase 1 best weights
│       │   └── vit_b16_final_best.pt           # Phase 2 best weights (final)
│       ├── figures/                             # loss curve, confusion matrices
│       ├── explainability/                      # all XAI output PNGs (13 figures)
│       │   ├── per_grade_samples.png
│       │   ├── gradcam_correct.png / gradcam_wrong.png
│       │   ├── gradcam_gallery.png
│       │   ├── attention_rollout.png
│       │   ├── grade1_false_negatives.png
│       │   ├── lime_correct.png / lime_wrong.png
│       │   ├── misclassification_overview.png
│       │   ├── worst_misclassifications.png
│       │   ├── per_class_metrics.png
│       │   ├── calibration_reliability.png
│       │   └── confidence_accuracy_tradeoff.png
│       └── results_log.csv                      # append-only run log
│
├── requirements.txt
├── Diabetic Retinopathy Grading.pdf       # coursework specification
└── README.md
```

---

## Notebooks

### `data_preprocessing.ipynb`
Loads the raw Kaggle train/test zips, performs exploratory data analysis (class distribution, resolution scatter, brightness by grade), applies Ben Graham's circular crop + contrast enhancement preprocessing, and writes the fixed 70/15/15 train/val/test CSV splits.

### `baseline.ipynb`
EfficientNet-B0 baseline trained from scratch with no class weighting or sampling. Establishes the starting QWK score against which all ViT experiments are benchmarked.

### `vit_b16_initial_experiments.ipynb`
Seven ablation experiments (1–8) building up to the final configuration:

| Exp | Change | Key finding |
|---|---|---|
| 1 | Class imbalance handling (CE vs WRS vs FocalLoss) | WRS + FocalLoss consistently outperforms plain CE |
| 2 | Enhanced augmentation pipeline | Flips + colour jitter + affine improve generalisation |
| 3 | Two-phase training (freeze → unfreeze) | Phase 1 head-only training stabilises convergence |
| 4 | LR × batch size × weight decay grid search | LR 1e-4 / 1e-6, batch 16, WD 3e-2 |
| 5 | Majority-class downsampling | Downsampling alone without WRS hurts severe grades |
| 6 | Two-phase + smart downsampling | Cap each class at Grade-1 count; best QWK in ablation |
| 8 | FocalLoss vs WRS ablation | FocalLoss(α=None, γ=2) + WRS chosen as final combination |

### `vit_b16_final.ipynb`
Final production notebook with 8 sections:

1. **Imports** — environment setup
2. **Configuration** — all hyperparameters in one cell (`DOWNSAMPLING`, `PHASE1_EPOCHS`, etc.)
3. **Load Data** — reads fixed CSV splits
4. **Dataset & DataLoaders** — `DRDataset`, augmentation pipeline, `WeightedRandomSampler`
5. **Model** — `vit_base_patch16_224.augreg_in21k_ft_in1k` via `timm`, 5-class head
6. **Training Utilities** — `run_epoch_clipped` with AMP + gradient clipping
7. **Final Model** — two-phase fine-tuning, QWK tracking, CSV logging, confusion matrices
8. **Explainability** — gradient-weighted attention (Grad-CAM), Attention Rollout, LIME, misclassification analysis, confidence calibration

---

## Model Architecture

| Setting | Value |
|---|---|
| Architecture | `vit_base_patch16_224.augreg_in21k_ft_in1k` (timm) |
| Pre-training | ImageNet-21k → ImageNet-1k fine-tune |
| Input size | 224 × 224, normalised to [−1, 1] |
| Head | Linear(768 → 5) replacing 1000-class head |
| Loss | FocalLoss(α=None, γ=2) |
| Imbalance | WeightedRandomSampler (+ optional smart downsampling) |
| Phase 1 | Frozen backbone, head only, LR 1e-4, LinearWarmup + CosineDecay |
| Phase 2 | All layers, LR 1e-6, gradient clip 1.0 |
| Metric | Quadratic Weighted Kappa (QWK) |

---

## Explainability Methods

All figures are saved to `outputs/vit_b16_final/explainability/` and conclusions are written inline in Section 8 of `vit_b16_final.ipynb`.