#!/usr/bin/env python3
"""Generate luminance for top2000 dataset - finds RAFs in DCIM subdirs."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import rawpy
from tqdm import tqdm

LUM_R, LUM_G, LUM_B = 0.2126, 0.7152, 0.0722
RAF_BASE = "/Volumes/4T/naoru/DCIM"
DATASET_DIR = "/Volumes/4T/xtrans-demosaic/datasets/top2000_dataset"
DOWNSAMPLE = 4


def find_raf(stem: str, base_dir: str) -> str | None:
    """Find RAF file for given stem in DCIM subdirs."""
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in [f"{stem}.RAF", f"{stem}.raf"]:
            fpath = os.path.join(subdir_path, fname)
            if os.path.exists(fpath):
                return fpath
    return None


def generate_luminance(raf_path: str) -> np.ndarray:
    """Generate luminance reference from RAF."""
    with rawpy.imread(raf_path) as raw:
        cfa = raw.raw_image_visible.astype(np.float32)
        black = raw.black_level_per_channel[0]
        white = raw.white_level
        cfa_norm = (cfa - black) / (white - black)
        cfa_norm = np.clip(cfa_norm, 0, 1)
        colors = raw.raw_colors_visible
    
    weights = np.where(colors == 0, LUM_R, 
              np.where(colors == 1, LUM_G, LUM_B)).astype(np.float32)
    weighted = cfa_norm * weights
    
    kernel_size = 3
    weight_sum = cv2.blur(weights, (kernel_size, kernel_size))
    lum = cv2.blur(weighted, (kernel_size, kernel_size)) / np.maximum(weight_sum, 1e-8)
    
    h, w = lum.shape
    new_h, new_w = h // DOWNSAMPLE, w // DOWNSAMPLE
    lum_down = cv2.resize(lum, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return lum_down.astype(np.float32)


def main():
    # Get list of stems from existing NPY files
    npy_files = [f for f in os.listdir(DATASET_DIR) 
                 if f.endswith('.npy') and not f.endswith('_lum.npy') and not f.endswith('_meta.npy')]
    stems = [os.path.splitext(f)[0] for f in npy_files]
    print(f"Found {len(stems)} NPY files in dataset")
    
    # Check which already have luminance
    existing_lum = set(f.replace('_lum.npy', '') for f in os.listdir(DATASET_DIR) if f.endswith('_lum.npy'))
    remaining = [s for s in stems if s not in existing_lum]
    print(f"Already have luminance: {len(existing_lum)}, remaining: {len(remaining)}")
    
    if not remaining:
        print("All done!")
        return
    
    # Process
    for stem in tqdm(remaining, desc="Generating luminance"):
        raf_path = find_raf(stem, RAF_BASE)
        if not raf_path:
            print(f"  {stem}: RAF not found, skipping")
            continue
        
        try:
            lum = generate_luminance(raf_path)
            out_path = os.path.join(DATASET_DIR, f"{stem}_lum.npy")
            np.save(out_path, lum)
        except Exception as e:
            print(f"  {stem}: ERROR - {e}")
    
    n_lum = len([f for f in os.listdir(DATASET_DIR) if f.endswith('_lum.npy')])
    print(f"\nDone! {n_lum} luminance files in {DATASET_DIR}")


if __name__ == '__main__':
    main()
