#!/usr/bin/env python3
"""
Build v4 training dataset from RAF files on NAS.

For each RAF:
1. DHT demosaic (rawpy, X-Trans native)
2. Black-subtract, normalize by (white - black), NO CLIP
3. Downscale 4x via area averaging
4. Save as float32 .npy

Output: one .npy per RAF in the output directory.
"""
import os
import sys
import time
import json
import subprocess
import gc
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import rawpy


NAS_USER = "naoru"
NAS_HOST = "192.168.1.101"
NAS_KEY = os.path.expanduser("~/.ssh/id_ed25519_nopass")
NAS_BASE = "/srv/dev-disk-by-uuid-0c04d244-f6e9-4730-bd25-b93ad683acfd/naoru/DCIM"

OUTPUT_DIR = "/Volumes/External/xtrans_v4_dataset"
DOWNSAMPLE = 4


def list_rafs_on_nas():
    """Get list of all RAF files on NAS."""
    cmd = [
        "ssh", "-i", NAS_KEY, "-o", "StrictHostKeyChecking=no",
        f"{NAS_USER}@{NAS_HOST}",
        f"find {NAS_BASE} -name '*.RAF' -o -name '*.raf' | sort"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"ERROR listing NAS: {result.stderr}")
        sys.exit(1)
    files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
    return files


def process_raf(args):
    """Process a single RAF file: fetch from NAS, demosaic, downsample, save."""
    nas_path, output_dir, index, total = args
    
    filename = os.path.basename(nas_path)
    stem = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{stem}.npy")
    
    # Skip if already processed
    if os.path.exists(output_path):
        return f"  [{index}/{total}] {stem}: already exists, skipping"
    
    try:
        # Fetch RAF from NAS to temp file
        with tempfile.NamedTemporaryFile(suffix='.RAF', delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = [
            "scp", "-i", NAS_KEY, "-o", "StrictHostKeyChecking=no",
            f"{NAS_USER}@{NAS_HOST}:{nas_path}", tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return f"  [{index}/{total}] {stem}: SCP failed"
        
        # Open with rawpy
        raw = rawpy.imread(tmp_path)
        
        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)
        
        # DHT demosaic, linear, no color matrix, no WB, no auto anything
        rgb_16 = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT,
            use_camera_wb=False,  # No WB - raw sensor values
            output_color=rawpy.ColorSpace.raw,  # No color matrix
            gamma=(1, 1),  # Linear
            output_bps=16,
            no_auto_bright=True,
            no_auto_scale=True,
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
        
        # Also save metadata
        meta = {
            'source': nas_path,
            'black_level': black,
            'white_level': white,
            'camera_wb': raw.camera_whitebalance[:3].tolist(),
            'original_size': [w, h],
            'downscaled_size': [new_w, new_h],
            'pattern': raw.raw_pattern.tolist(),
            'range_min': float(downscaled.min()),
            'range_max': float(downscaled.max()),
        }
        
        meta_path = os.path.join(output_dir, f"{stem}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        
        raw.close()
        
        # Cleanup temp
        os.unlink(tmp_path)
        
        return f"  [{index}/{total}] {stem}: {new_w}x{new_h} range=[{downscaled.min():.3f}, {downscaled.max():.3f}]"
    
    except Exception as e:
        # Cleanup temp on error
        try:
            os.unlink(tmp_path)
        except:
            pass
        return f"  [{index}/{total}] {stem}: ERROR - {e}"


def main():
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("Listing RAF files on NAS...")
    raf_files = list_rafs_on_nas()
    print(f"Found {len(raf_files)} RAF files")
    
    # Check how many already done
    existing = set(f.replace('.npy', '') for f in os.listdir(output_dir) if f.endswith('.npy'))
    remaining = [f for f in raf_files if os.path.splitext(os.path.basename(f))[0] not in existing]
    print(f"Already processed: {len(existing)}, remaining: {len(remaining)}")
    
    # Prepare args
    args = [
        (path, output_dir, i+1, len(raf_files))
        for i, path in enumerate(raf_files)
    ]
    
    # Process with limited parallelism (NAS I/O + memory bounded)
    n_workers = min(4, cpu_count())
    print(f"Processing with {n_workers} workers...")
    t0 = time.time()
    
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(process_raf, args):
            print(result)
    
    elapsed = time.time() - t0
    
    # Summary
    n_done = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir) if f.endswith('.npy')
    )
    
    print(f"\nDone! {n_done} files in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Total size: {total_size / 1e9:.1f} GB")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
