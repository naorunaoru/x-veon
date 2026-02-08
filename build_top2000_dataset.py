#!/usr/bin/env python3
"""
Build training dataset from top2000_sharp.json using local 4T drive.

For each RAF:
1. DHT demosaic (rawpy, X-Trans native)
2. Black-subtract, normalize by (white - black), NO CLIP
3. Downscale 4x via area averaging
4. Save as float32 .npy
"""
import os
import sys
import time
import json
import gc
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import rawpy


# Config
RAF_BASE = "/Volumes/4T/naoru/DCIM"
TOP2000_JSON = "data/top2000_sharp.json"
OUTPUT_DIR = "/Volumes/4T/xtrans-demosaic/datasets/top2000_dataset"
DOWNSAMPLE = 4


def find_raf_files(image_ids: list[str], base_dir: str) -> dict[str, str]:
    """Find RAF files for given image IDs in the DCIM subfolders."""
    found = {}
    
    # Walk through all subdirs looking for RAF files
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        for filename in os.listdir(subdir_path):
            if not filename.upper().endswith('.RAF'):
                continue
            stem = os.path.splitext(filename)[0]
            if stem in image_ids:
                found[stem] = os.path.join(subdir_path, filename)
    
    return found


def process_raf(args):
    """Process a single RAF file: demosaic, downsample, save."""
    raf_path, output_dir, stem, index, total = args
    
    output_path = os.path.join(output_dir, f"{stem}.npy")
    
    # Skip if already processed
    if os.path.exists(output_path):
        return f"  [{index}/{total}] {stem}: exists, skipping"
    
    try:
        # Open with rawpy
        raw = rawpy.imread(raf_path)
        
        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)
        
        # Demosaic using DHT (X-Trans native algorithm)
        rgb_16 = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT,
            output_bps=16,
            no_auto_bright=True,
            no_auto_scale=True,  # CRITICAL: keeps values in raw range, not scaled to 16-bit
            gamma=(1, 1),  # Linear
            output_color=rawpy.ColorSpace.raw,  # No color matrix
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=[1, 1, 1, 1],  # Unity WB
            user_flip=0,  # No EXIF rotation - keep raw sensor orientation
        )
        
        h, w = rgb_16.shape[:2]
        
        # Normalize: subtract black, divide by (white - black), NO CLIP
        rgb_f = (rgb_16.astype(np.float32) - black) / (white - black)
        del rgb_16
        
        # Downscale 4x via area averaging
        new_h, new_w = h // DOWNSAMPLE, w // DOWNSAMPLE
        h_crop = new_h * DOWNSAMPLE
        w_crop = new_w * DOWNSAMPLE
        
        downscaled = np.zeros((new_h, new_w, 3), dtype=np.float32)
        for c in range(3):
            ch = rgb_f[:h_crop, :w_crop, c].reshape(new_h, DOWNSAMPLE, new_w, DOWNSAMPLE)
            downscaled[:, :, c] = ch.mean(axis=(1, 3))
        
        del rgb_f
        gc.collect()
        
        # Save
        np.save(output_path, downscaled)
        
        # Also save metadata (use list() for compatibility with numpy arrays and lists)
        meta = {
            'source': raf_path,
            'black_level': black,
            'white_level': white,
            'camera_wb': list(raw.camera_whitebalance[:3]),
            'original_size': [w, h],
            'downscaled_size': [new_w, new_h],
            'pattern': [list(row) for row in raw.raw_pattern],
            'range_min': float(downscaled.min()),
            'range_max': float(downscaled.max()),
        }
        
        meta_path = os.path.join(output_dir, f"{stem}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        
        raw.close()
        
        return f"  [{index}/{total}] {stem}: {new_w}x{new_h} range=[{downscaled.min():.3f}, {downscaled.max():.3f}]"
    
    except Exception as e:
        return f"  [{index}/{total}] {stem}: ERROR - {e}"


def main():
    # Load top2000 list
    with open(TOP2000_JSON) as f:
        image_ids = set(json.load(f))
    print(f"Loaded {len(image_ids)} image IDs from {TOP2000_JSON}")
    
    # Find RAF files
    print(f"Searching for RAFs in {RAF_BASE}...")
    raf_map = find_raf_files(image_ids, RAF_BASE)
    print(f"Found {len(raf_map)}/{len(image_ids)} RAF files")
    
    missing = image_ids - set(raf_map.keys())
    if missing:
        print(f"Missing {len(missing)} files: {list(missing)[:10]}...")
    
    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check how many already done
    existing = set(f.replace('.npy', '') for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy') and not f.endswith('_lum.npy'))
    remaining = {k: v for k, v in raf_map.items() if k not in existing}
    print(f"Already processed: {len(existing)}, remaining: {len(remaining)}")
    
    if not remaining:
        print("Nothing to do!")
        return
    
    # Prepare args
    args = [
        (path, OUTPUT_DIR, stem, i+1, len(remaining))
        for i, (stem, path) in enumerate(remaining.items())
    ]
    
    # Process with limited parallelism (local SSD, but memory bounded)
    n_workers = min(4, cpu_count())
    print(f"Processing with {n_workers} workers...")
    t0 = time.time()
    
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(process_raf, args):
            print(result)
    
    elapsed = time.time() - t0
    
    # Summary
    n_done = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy') and not f.endswith('_lum.npy')])
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')
    )
    
    print(f"\nDone! {n_done} files in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Total size: {total_size / 1e9:.1f} GB")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
