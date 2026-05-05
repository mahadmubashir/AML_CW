"""
RetinaDataset and fundus preprocessing helpers.

# lives in a real module file so DataLoader workers spawned via
# multiprocessing 'spawn' can import the class — notebook cells
# can't define spawn-compatible classes
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
# tolerate the few truncated jpegs in the Kaggle DR dataset (33 bytes
# missing on a handful of files) — PIL fills the gap, training continues
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage.filters import frangi
from skimage.restoration import denoise_bilateral
from torch.utils.data import Dataset


def preprocess_fundus_pil(image: Image.Image, steps: dict) -> Image.Image:
    """Best-effort approximation of the reference repo's preprocessing stages."""
    # PIL -> numpy RGB -> OpenCV BGR for the cv2 ops below
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if steps.get("crop_black", True):
        # crops the black border around the fundus circle
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img_bgr = img_bgr[y : y + h, x : x + w]

    if steps.get("circular_mask", True):
        # zero out everything outside the eyeball circle
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        r = int(0.48 * min(h, w))
        cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    if steps.get("glare_inpaint", True):
        # find bright glare spots and inpaint over them
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        glare = (v > 245).astype(np.uint8) * 255
        if glare.sum() > 0:
            glare = cv2.dilate(glare, np.ones((3, 3), np.uint8), iterations=1)
            img_bgr = cv2.inpaint(img_bgr, glare, 3, cv2.INPAINT_TELEA)

    if steps.get("ben_graham", True):
        # Ben Graham's contrast trick from Kaggle 2015 winner
        blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=10)
        img_bgr = cv2.addWeighted(img_bgr, 4, blur, -4, 128)

    if steps.get("clahe", True):
        # local contrast on the L channel of LAB color space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        img_bgr = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    if steps.get("green_enhance", True):
        # green channel holds most retinal lesion detail
        b, g, r = cv2.split(img_bgr)
        g = cv2.equalizeHist(g)
        img_bgr = cv2.merge((b, g, r))

    if steps.get("vessel_frangi", False):
        # Frangi vesselness filter — blends a vessel map back in
        green = cv2.split(img_bgr)[1].astype(np.float32) / 255.0
        vessels = frangi(green)
        vessels = (255 * (vessels / (vessels.max() + 1e-6))).astype(np.uint8)
        vessels_bgr = cv2.cvtColor(vessels, cv2.COLOR_GRAY2BGR)
        img_bgr = cv2.addWeighted(img_bgr, 0.8, vessels_bgr, 0.2, 0)

    if steps.get("red_enhance", False):
        # red channel for hemorrhage spots
        b, g, r = cv2.split(img_bgr)
        r = cv2.equalizeHist(r)
        img_bgr = cv2.merge((b, g, r))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if steps.get("adaptive_denoise", True):
        # smooths noise while keeping edges sharp
        img_dn = denoise_bilateral(img_rgb, sigma_color=0.05, sigma_spatial=3, channel_axis=-1)
        img_rgb = (np.clip(img_dn, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(img_rgb)


class RetinaDataset(Dataset):
    """Fundus images + integer labels 0..NUM_CLASSES-1."""

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root,
        image_extension: str,
        transform,
        preprocess: bool,
        preprocess_steps: dict | None = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.data_root = Path(data_root)
        # accept ".jpeg" or "jpeg" — normalise to leading dot
        ext = image_extension if str(image_extension).startswith(".") else f".{image_extension}"
        self.image_extension = ext
        self.transform = transform
        self.preprocess = bool(preprocess)
        # carrying preprocess steps explicitly so spawned workers don't need
        # access to a notebook-level PREPROCESS global
        self.preprocess_steps = preprocess_steps or {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        stem = str(row["image"])
        # rows can be either bare ids or already include extension — handle both
        if not stem.lower().endswith((".jpeg", ".jpg", ".png")):
            stem = stem + self.image_extension
        path = self.data_root / stem
        image = Image.open(path).convert("RGB")
        if self.preprocess:
            image = preprocess_fundus_pil(image, self.preprocess_steps)
        label = int(row["level"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label
