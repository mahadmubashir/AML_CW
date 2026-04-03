# AML_G14_CW — Diabetic Retinopathy Grading (EEEM068)

Coursework repository for **Applied Machine Learning**: train CNN / ViT models on fundus images for **5-class severity grading** (0–4).

## Quick start

1. **Python environment** (3.10+ recommended):

   ```bash
   cd AML_G14_CW
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Install PyTorch with GPU support from [pytorch.org](https://pytorch.org) if needed.

2. **Dataset (Kaggle)**

   Download from the [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) competition:

   - Unzip **`train.zip`** so JPEGs live under `data/train/` (or set `paths.data_root` in your config).
   - Place **`trainLabels.csv`** at `data/trainLabels.csv` (columns: `image`, `level`).

   Expected layout:

   ```text
   data/
   ├── train/
   │   ├── 10_left.jpeg
   │   └── ...
   └── trainLabels.csv
   ```

3. **Run training**

   Open [`notebooks/dr_training.ipynb`](notebooks/dr_training.ipynb). Set **`CONFIG_PATH`** (first configuration cell) to your YAML file under `configs/`, then **Run All**.

   Each group member should copy an example config to `configs/member_<yourname>.yaml`, adjust hyperparameters and `model.architecture`, and point **`CONFIG_PATH`** to that file—no need to edit training code.

## Repository layout

- `configs/` — Experiment YAML files (`paths`, `model`, `training`, `augmentation`, `loss`, …).
- `notebooks/dr_training.ipynb` — Single notebook: loader, split, augmentations, model, train/val, metrics (including quadratic weighted kappa).
- `data/` — Local dataset (gitignored except `.gitkeep`).
- `checkpoints/` — Saved weights (gitignored).
- `outputs/` — Plots / submission CSVs (gitignored).

## Config workflow (five collaborators)

1. Duplicate `configs/member_1.yaml` → `configs/member_<you>.yaml`.
2. Change only YAML fields: `experiment_name`, `model.architecture`, `training.*`, `augmentation.*`, etc.
3. In the notebook, set `CONFIG_PATH = "configs/member_<you>.yaml"` (path relative to repository root).
4. Commit the YAML file; avoid merge conflicts by not editing the same notebook lines—only change `CONFIG_PATH` locally if needed.

### Supported `model.architecture` values (torchvision)

These names match `torchvision.models` constructors directly:

- **EfficientNet:** `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7`
- **ViT:** `vit_b_16`, `vit_l_16`, `vit_h_14` (use an input size appropriate to the variant; `vit_h_14` often expects a larger `data.image_size`—see torchvision docs)

If your label CSV uses different column names, normalize them to **`image`** and **`level`** (0–4), or adjust the notebook where the CSV is read.
