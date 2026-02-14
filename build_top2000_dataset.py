#!/usr/bin/env python3
"""
Build training dataset from a ranking/classification JSON file.

Supported JSON formats:
  - hf_ha_ranking.json:     [{"filename": "DSCF4582", "path": "...", ...}, ...]
  - raf_classification.json: [{"file": "DSCF7016.RAF", "path": "...", ...}, ...]
  - top2000_sharp.json:      ["DSCF7140", ...] (legacy, needs --raf-base)

For each RAF:
1. DHT demosaic (rawpy, X-Trans native)
2. Black-subtract, normalize by (white - black), NO CLIP
3. Downscale 4x via area averaging
4. Save as float32 .npy
"""
import argparse
import os
import sys
import time
import json
import gc
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import rawpy


DOWNSAMPLE = 4


def load_raf_map(json_path: str, raf_base: str | None = None, top_n: int | None = None) -> dict[str, str]:
    """Load a JSON ranking file and return {stem: raf_path} dict."""
    with open(json_path) as f:
        data = json.load(f)

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        # hf_ha_ranking.json or raf_classification.json
        raf_map = {}
        for entry in data:
            path = entry["path"]
            if "filename" in entry:
                stem = entry["filename"]
            else:
                stem = os.path.splitext(entry["file"])[0]
            raf_map[stem] = path
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        # Legacy: plain list of stems, needs raf_base
        if not raf_base:
            sys.exit("Legacy JSON (plain list of IDs) requires --raf-base")
        image_ids = set(data)
        raf_map = {}
        for subdir in os.listdir(raf_base):
            subdir_path = os.path.join(raf_base, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for filename in os.listdir(subdir_path):
                if not filename.upper().endswith('.RAF'):
                    continue
                stem = os.path.splitext(filename)[0]
                if stem in image_ids:
                    raf_map[stem] = os.path.join(subdir_path, filename)
    else:
        sys.exit(f"Unrecognized JSON format in {json_path}")

    if top_n and top_n < len(raf_map):
        # Preserve order from the JSON (assumed pre-sorted by quality)
        raf_map = dict(list(raf_map.items())[:top_n])

    return raf_map


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
            'pattern': [[int(v) for v in row] for row in raw.raw_pattern],
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
    parser = argparse.ArgumentParser(description="Build demosaic training dataset from RAF ranking JSON")
    parser.add_argument("json_file", help="Ranking JSON (hf_ha_ranking.json, raf_classification.json, or legacy ID list)")
    parser.add_argument("-o", "--output", required=True, help="Output directory for .npy files")
    parser.add_argument("-n", "--top-n", type=int, default=None, help="Only process the top N entries (default: all)")
    parser.add_argument("--raf-base", default=None, help="Base DCIM directory (only needed for legacy plain-list JSON)")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    # Load RAF map from JSON
    raf_map = load_raf_map(args.json_file, raf_base=args.raf_base, top_n=args.top_n)
    print(f"Loaded {len(raf_map)} RAFs from {args.json_file}")

    # Verify paths exist
    missing = {k: v for k, v in raf_map.items() if not os.path.exists(v)}
    if missing:
        print(f"WARNING: {len(missing)} RAF files not found on disk (skipping)")
        for stem in list(missing)[:5]:
            print(f"  {missing[stem]}")
        raf_map = {k: v for k, v in raf_map.items() if k not in missing}

    # Create output dir
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Check how many already done
    existing = set(f.replace('.npy', '') for f in os.listdir(output_dir) if f.endswith('.npy') and not f.endswith('_lum.npy'))
    remaining = {k: v for k, v in raf_map.items() if k not in existing}
    print(f"Already processed: {len(existing)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return

    # Prepare args
    work_args = [
        (path, output_dir, stem, i+1, len(remaining))
        for i, (stem, path) in enumerate(remaining.items())
    ]

    # Process with limited parallelism (memory bounded)
    n_workers = min(args.workers, cpu_count())
    print(f"Processing with {n_workers} workers...")
    t0 = time.time()

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(process_raf, work_args):
            print(result)

    elapsed = time.time() - t0

    # Summary
    n_done = len([f for f in os.listdir(output_dir) if f.endswith('.npy') and not f.endswith('_lum.npy')])
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir) if f.endswith('.npy')
    )

    print(f"\nDone! {n_done} files in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Total size: {total_size / 1e9:.1f} GB")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
