#!/usr/bin/env python
"""
build_preprocess_cache.py — pre-compute preprocessed fundus images.

reads every .jpeg in SRC_DIR, runs the same preprocessing pipeline used by
src/retina_dataset.py, writes the result as JPEG into DST_DIR. uses a
multiprocessing pool to fan out across all available cores.

run once. after this finishes, training with USE_PREPROCESSING=1 just reads
the cached files instead of redoing cv2 work every batch.
"""

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

# adding repo root so the existing preprocess_fundus_pil import works
REPO_ROOT = Path("/users/na02153/AML_G14_CW")
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402
from src.retina_dataset import preprocess_fundus_pil  # noqa: E402

# matches the PREPROCESS dict in train_script.py
PREPROCESS_STEPS = dict(
    crop_black=True,
    circular_mask=True,
    glare_inpaint=True,
    ben_graham=True,
    clahe=True,
    green_enhance=True,
    vessel_frangi=False,
    red_enhance=False,
    adaptive_denoise=True,
)

SRC_DIR = Path(os.environ.get("DR_SRC_DIR", "./data/train"))
DST_DIR = Path(os.environ.get("DR_DST_DIR", "./data/train_preprocessed"))
JPEG_QUALITY = 95


def process_one(src_path_str):
    """worker function — preprocess one image and save it. returns (filename, ok, msg)."""
    src_path = Path(src_path_str)
    dst_path = DST_DIR / src_path.name
    # skip if already exists — makes the script resumable on partial runs
    if dst_path.exists():
        return (src_path.name, True, "skipped")
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            out = preprocess_fundus_pil(im, PREPROCESS_STEPS)
            out.save(dst_path, format="JPEG", quality=JPEG_QUALITY)
        return (src_path.name, True, "ok")
    except Exception as e:
        return (src_path.name, False, f"error: {e!r}")


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # gathering every jpeg path up front so the pool can chunk them
    src_files = sorted(str(p) for p in SRC_DIR.glob("*.jpeg"))
    total = len(src_files)
    print(f"found {total} source images")
    print(f"src: {SRC_DIR}")
    print(f"dst: {DST_DIR}")

    # using one less worker than core count to leave headroom for the OS
    n_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"spawning {n_workers} workers")

    t0 = time.time()
    ok_count = 0
    skip_count = 0
    err_count = 0
    err_samples = []

    # imap_unordered = workers stream results back as they finish, no need to wait
    with mp.Pool(processes=n_workers) as pool:
        for i, (name, ok, msg) in enumerate(
            pool.imap_unordered(process_one, src_files, chunksize=8), start=1
        ):
            if ok and msg == "ok":
                ok_count += 1
            elif ok and msg == "skipped":
                skip_count += 1
            else:
                err_count += 1
                if len(err_samples) < 10:
                    err_samples.append(f"{name}: {msg}")
            # printing progress every 500 images so the log shows movement
            if i % 500 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta_sec = (total - i) / max(rate, 1e-6)
                print(
                    f"[{i}/{total}] ok={ok_count} skipped={skip_count} "
                    f"err={err_count} | {rate:.1f} img/s | eta {eta_sec/60:.1f} min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print("---")
    print(f"done in {elapsed/60:.1f} min")
    print(f"ok={ok_count} skipped={skip_count} err={err_count}")
    if err_samples:
        print("first errors:")
        for s in err_samples:
            print(f"  {s}")


if __name__ == "__main__":
    main()
