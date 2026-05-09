"""
utils.py — Shared infrastructure for SwinV2-Small DR grading experiments.
All experiment notebooks do:   from utils import *
Place this file in the same folder as all experiment notebooks.
"""
import os, random
from pathlib import Path

import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import (
    classification_report, cohen_kappa_score, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ──────────────────────────────────────────────────────────────────────
NOTEBOOK_DIR    = Path(__file__).resolve().parent
ROOT            = NOTEBOOK_DIR.parent
DATA_ROOT       = ROOT / 'data' / 'diabetic-retinopathy-detection'
TRAIN_DIR       = DATA_ROOT / 'train_proc'
VAL_DIR         = DATA_ROOT / 'val_proc'
TEST_LABELS_DIR = DATA_ROOT / 'test_labels_proc'
TRAIN_CSV       = DATA_ROOT / 'train_split.csv'
VAL_CSV         = DATA_ROOT / 'val_split.csv'
TEST_CSV        = DATA_ROOT / 'test_split.csv'
WEIGHTS_PATH    = ROOT / 'weights' / 'swinv2_small.safetensors'

# ── Constants ─────────────────────────────────────────────────────────────────
SEED        = 42
NUM_CLASSES = 5
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
IMAGE_SIZE  = 256
SWIN_MEAN   = (0.485, 0.456, 0.406)
SWIN_STD    = (0.229, 0.224, 0.225)
MODEL_NAME  = 'swinv2_small_window8_256'

# ── Training defaults (override in each notebook if needed) ───────────────────
BATCH_SIZE              = 16
EPOCHS                  = 20
LR                      = 5e-5
WEIGHT_DECAY            = 3e-2
NUM_WORKERS             = 4
EARLY_STOPPING_PATIENCE = 6
DROPOUT                 = 0.3

# ── Seed + device ─────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset ───────────────────────────────────────────────────────────────────
class DRDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = Path(data_dir)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(self.data_dir / f"{row['image']}.jpeg").convert('RGB')
        label = int(row['level'])
        if self.transform: image = self.transform(image)
        return image, label

# ── Transforms ────────────────────────────────────────────────────────────────
baseline_train_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(SWIN_MEAN, SWIN_STD),
])
val_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.05)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(SWIN_MEAN, SWIN_STD),
])
enhanced_train_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.05)], p=0.4),
    transforms.RandomApply([transforms.RandomAffine(0,translate=(0.05,0.05),scale=(0.95,1.05))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(SWIN_MEAN, SWIN_STD),
    transforms.RandomErasing(p=0.2),
])

# ── Load splits ───────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

# ── Class weights ─────────────────────────────────────────────────────────────
class_weights_np = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES),
                                         y=train_df['level'].values)
class_weights_t  = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
sample_weights_t = torch.as_tensor(class_weights_np[train_df['level'].values], dtype=torch.double)
wrs_sampler      = WeightedRandomSampler(sample_weights_t, len(sample_weights_t), replacement=True)

# ── FocalLoss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()

# ── Model builder ─────────────────────────────────────────────────────────────
def _load_swin_weights(backbone, weights_path):
    from timm.models._helpers import load_checkpoint
    r = load_checkpoint(backbone, weights_path, strict=False)
    n_h = len([k for k in r.missing_keys    if 'custom_head' in k])
    n_s = len([k for k in r.unexpected_keys if 'head.fc' in k or 'attn_mask' in k])
    print(f'  Weights loaded — head keys (random init): {n_h}, old keys ignored: {n_s}')

def build_swin_model(pretrained=True):
    backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, global_pool='avg')
    feat_dim = backbone.num_features
    backbone.custom_head = nn.Sequential(
        nn.LayerNorm(feat_dim), nn.Dropout(p=DROPOUT), nn.Linear(feat_dim, NUM_CLASSES))
    _base = backbone.forward
    backbone.forward = lambda x: backbone.custom_head(_base(x))
    wp = Path(str(WEIGHTS_PATH)) if WEIGHTS_PATH else None
    if wp and wp.exists():
        print(f'Loading from local file: {wp.name}')
        _load_swin_weights(backbone, str(wp))
    elif pretrained:
        try:
            print('Downloading from HuggingFace...')
            tmp = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, global_pool='avg')
            own = backbone.state_dict()
            src = {k:v for k,v in tmp.state_dict().items() if k in own and 'custom_head' not in k}
            own.update(src); backbone.load_state_dict(own, strict=False); del tmp
            print('Downloaded and loaded.')
        except Exception as e:
            print(f'Download failed: {e}'); raise
    else:
        print('Random init.')
    print(f'SwinV2-Small | {sum(p.numel() for p in backbone.parameters())/1e6:.1f}M params')
    return backbone

# ── run_epoch ─────────────────────────────────────────────────────────────────
# NOTE: uses module-level model/criterion/optimizer/scaler — set these before calling
use_amp = device.type == 'cuda'

def run_epoch(loader, train_mode, model, criterion, optimizer, scaler, grad_clip=1.0):
    model.train(train_mode)
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, leave=False, desc='train' if train_mode else 'val  '):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        total_loss += loss.item() * labels.size(0); total += labels.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss/total, all_labels, all_preds, cohen_kappa_score(all_labels, all_preds, weights='quadratic')

# ── Standard train loop ───────────────────────────────────────────────────────
def train_loop(model, criterion, optimizer, scheduler, scaler,
               train_loader, val_loader, ckpt_path,
               epochs=EPOCHS, patience=EARLY_STOPPING_PATIENCE):
    best_kappa, pat_left = -1.0, patience
    history = {'train_loss':[],'val_loss':[],'train_kappa':[],'val_kappa':[]}
    print(f"{'Epoch':<8} {'Tr Loss':>10} {'Tr κ':>8} {'Val Loss':>10} {'Val κ':>8}")
    print('─' * 52)
    for epoch in range(1, epochs+1):
        tl,_,_,tk = run_epoch(train_loader,True, model,criterion,optimizer,scaler)
        vl,vl_y,vp,vk = run_epoch(val_loader,False,model,criterion,optimizer,scaler)
        scheduler.step()
        for k,v in [('train_loss',tl),('val_loss',vl),('train_kappa',tk),('val_kappa',vk)]:
            history[k].append(v)
        marker = ''
        if vk > best_kappa:
            best_kappa = vk; pat_left = patience
            torch.save({'epoch':epoch,'model_state':model.state_dict(),
                        'val_kappa':vk,'val_loss':vl}, ckpt_path)
            marker = '  ★'
        print(f'{epoch:<8} {tl:>10.4f} {tk:>8.4f} {vl:>10.4f} {vk:>8.4f}{marker}')
        if not marker:
            pat_left -= 1
            if pat_left <= 0: print(f'Early stopping at epoch {epoch}.'); break
    print(f'\nBest val κ = {best_kappa:.4f}')
    return history, best_kappa

# ── Eval helper ───────────────────────────────────────────────────────────────
def evaluate(model, criterion, optimizer, scaler, loader, ckpt_path, label):
    m = build_swin_model(pretrained=False).to(device)
    c = nn.CrossEntropyLoss(label_smoothing=0.1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt['model_state'])
    print(f'Loaded {Path(ckpt_path).name}  (epoch {ckpt["epoch"]}, κ={ckpt["val_kappa"]:.4f})')
    _,y_true,y_pred,_ = run_epoch(loader, False, m, c, None, None)
    print(f'\n{label} — Test Set:')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))
    print(f'QWK: {cohen_kappa_score(y_true, y_pred, weights="quadratic"):.4f}')
    return y_true, y_pred

def plot_confusion(y_true, y_pred, title, fig_path, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float)/cm.sum(axis=1,keepdims=True)
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    for ax,data,t,fmt in zip(axes,[cm,cm_n],[title+' counts',title+' normalised'],['d','.2f']):
        im=ax.imshow(data,cmap=cmap,vmin=None if fmt=='d' else 0,vmax=None if fmt=='d' else 1)
        fig.colorbar(im,ax=ax,format=None if fmt=='d' else '%.2f')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,
               xlabel='Predicted',ylabel='True',title=t)
        plt.setp(ax.get_xticklabels(),rotation=30,ha='right')
        thresh=(data.max()/2) if fmt=='d' else 0.5
        for i in range(5):
            for j in range(5):
                ax.text(j,i,format(data[i,j],fmt),ha='center',va='center',fontsize=9,
                        color='white' if data[i,j]>thresh else 'black')
    plt.tight_layout(); fig.savefig(fig_path,dpi=150,bbox_inches='tight'); plt.show()

def make_loader(ds, sampler=None, shuffle=True):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(shuffle and sampler is None),
                      sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True,
                      drop_last=True, persistent_workers=NUM_WORKERS>0)

def make_val_loader(ds):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS>0)

def layer_wise_optimizer(model, lr=LR, wd=WEIGHT_DECAY):
    bp = [p for n,p in model.named_parameters() if 'custom_head' not in n and p.requires_grad]
    hp = [p for n,p in model.named_parameters() if 'custom_head'     in n and p.requires_grad]
    return torch.optim.AdamW([{'params':bp,'lr':lr*0.1},{'params':hp,'lr':lr}], weight_decay=wd)

print(f'utils.py loaded — device={device}, train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')
