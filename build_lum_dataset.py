#!/usr/bin/env python3
"""
Generate luminance reference channel for v4.2 training.

Takes RAF files, creates weighted monochrome (luminance coefficients per CFA position),
downsamples 4x to match RGB targets, saves as *_lum.npy.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import rawpy
from tqdm import tqdm


# Luminance coefficients (Rec. 709)
LUM_R, LUM_G, LUM_B = 0.2126, 0.7152, 0.0722


def generate_luminance_reference(raf_path: str, downsample: int = 4) -> np.ndarray:
    """
    Generate luminance reference from RAF file.
    
    1. Load raw CFA
    2. Weight each pixel by luminance coefficient based on its color
    3. Blur to interpolate
    4. Downsample to match RGB target resolution
    """
    with rawpy.imread(raf_path) as raw:
        cfa = raw.raw_image_visible.astype(np.float32)
        black = raw.black_level_per_channel[0]
        white = raw.white_level
        cfa_norm = (cfa - black) / (white - black)
        cfa_norm = np.clip(cfa_norm, 0, 1)
        
        # Get color at each position (0=R, 1=G, 2=B)
        colors = raw.raw_colors_visible
    
    # Weight by luminance coefficients
    weights = np.where(colors == 0, LUM_R, 
              np.where(colors == 1, LUM_G, LUM_B)).astype(np.float32)
    
    weighted = cfa_norm * weights
    
    # Interpolate: blur then normalize by weight sum
    kernel_size = 3
    weight_sum = cv2.blur(weights, (kernel_size, kernel_size))
    lum = cv2.blur(weighted, (kernel_size, kernel_size)) / np.maximum(weight_sum, 1e-8)
    
    # Downsample 4x to match RGB targets
    h, w = lum.shape
    new_h, new_w = h // downsample, w // downsample
    lum_down = cv2.resize(lum, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return lum_down.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raf-dir", type=str, required=True,
                        help="Directory containing RAF files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save *_lum.npy files")
    parser.add_argument("--filter-file", type=str, default=None,
                        help="JSON file with list of stems to process")
    parser.add_argument("--downsample", type=int, default=4)
    args = parser.parse_args()
    
    raf_dir = Path(args.raf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find RAF files
    raf_files = list(raf_dir.glob("*.RAF")) + list(raf_dir.glob("*.raf"))
    print(f"Found {len(raf_files)} RAF files")
    
    # Filter if specified
    if args.filter_file:
        with open(args.filter_file) as f:
            allowed = set(json.load(f))
        raf_files = [r for r in raf_files if r.stem in allowed]
        print(f"Filtered to {len(raf_files)} files")
    
    # Process
    for raf_path in tqdm(raf_files, desc="Generating luminance"):
        output_path = output_dir / f"{raf_path.stem}_lum.npy"
        
        if output_path.exists():
            continue
        
        try:
            lum = generate_luminance_reference(str(raf_path), args.downsample)
            np.save(output_path, lum)
        except Exception as e:
            print(f"Error processing {raf_path.name}: {e}")
    
    print(f"Done! Saved to {output_dir}")


if __name__ == "__main__":
    main()
