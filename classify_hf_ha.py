#!/usr/bin/env python3
"""
Classify RAF images by high-frequency high-amplitude content.

Pipeline per image:
1. rawpy DHT demosaic, camera WB, linear (no gamma), no color matrix
2. Black-subtract, normalize by (white - black)
3. Compute luminance = 0.2126*R + 0.7152*G + 0.0722*B (linear)
4. Sobel gradient magnitude on luminance
5. Metric = 99th percentile of (gradient * luminance)  — "bright edges"
6. Also record max pixel value across channels

Output: JSON sorted by metric descending.
"""
import os
import sys
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import rawpy


RAF_BASE = "/Volumes/4T/naoru/DCIM"
OUTPUT_JSON = "/Volumes/4T/xtrans-demosaic/datasets/hf_ha_ranking.json"


def sobel_magnitude(img):
    """Compute Sobel gradient magnitude on a 2D array."""
    # Sobel kernels via finite differences (fast, no scipy needed)
    # Gx = img[r, c+1] - img[r, c-1], Gy = img[r+1, c] - img[r-1, c]
    gx = img[:, 2:] - img[:, :-2]
    gy = img[2:, :] - img[:-2, :]
    # Trim to common size
    gx = gx[1:-1, :]
    gy = gy[:, 1:-1]
    return np.sqrt(gx**2 + gy**2)


def process_raf(args):
    """Process one RAF, return dict with metrics or None on error."""
    raf_path, index, total = args

    stem = Path(raf_path).stem
    folder = Path(raf_path).parent.name

    try:
        raw = rawpy.imread(raf_path)

        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)
        scale = white - black
        if scale <= 0:
            return None

        # DHT demosaic, camera WB, linear, raw color space
        rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT,
            output_bps=16,
            no_auto_bright=True,
            no_auto_scale=True,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.raw,
            use_camera_wb=True,
            use_auto_wb=False,
            user_flip=0,
        ).astype(np.float32)

        raw.close()

        # Normalize
        rgb = (rgb - black) / scale

        max_val = float(np.max(rgb))

        # Luminance (approximate, raw color space but close enough for ranking)
        lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

        # Gradient magnitude
        grad = sobel_magnitude(lum)

        # Trim luminance to match grad shape
        lum_trimmed = lum[1:-1, 1:-1]

        # Metric: gradient * intensity, take 99th percentile
        product = grad * lum_trimmed
        metric = float(np.percentile(product, 99))

        # Also grab 99.9th for extreme highlights
        metric_999 = float(np.percentile(product, 99.9))

        if index % 50 == 0:
            print(f"  [{index}/{total}] {stem}: metric={metric:.4f}, max={max_val:.3f}", flush=True)

        return {
            "filename": stem,
            "path": raf_path,
            "folder": folder,
            "max_range": round(max_val, 4),
            "hf_ha_p99": round(metric, 6),
            "hf_ha_p999": round(metric_999, 6),
        }

    except Exception as e:
        print(f"  [{index}/{total}] {stem}: ERROR {e}", flush=True)
        return None


def main():
    # Find all RAF files (skip ._ macOS resource forks)
    print(f"Scanning {RAF_BASE} for RAF files...")
    raf_files = []
    for root, dirs, files in os.walk(RAF_BASE):
        for f in sorted(files):
            if f.upper().endswith('.RAF') and not f.startswith('._'):
                raf_files.append(os.path.join(root, f))

    raf_files.sort()
    total = len(raf_files)
    print(f"Found {total} RAF files")

    # Build args
    args = [(path, i, total) for i, path in enumerate(raf_files)]

    # Process with multiprocessing
    ncpu = cpu_count()
    print(f"Processing with {ncpu} workers...")
    t0 = time.time()

    results = []
    with Pool(ncpu) as pool:
        for result in pool.imap_unordered(process_raf, args, chunksize=4):
            if result is not None:
                results.append(result)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(results)}/{total} images in {elapsed:.1f}s ({elapsed/total:.2f}s/img)")

    # Sort by metric descending
    results.sort(key=lambda x: x["hf_ha_p99"], reverse=True)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUTPUT_JSON}")

    # Print top 20
    print("\nTop 20 high-frequency high-amplitude images:")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:3d}. {r['filename']:20s}  p99={r['hf_ha_p99']:.4f}  p999={r['hf_ha_p999']:.4f}  max={r['max_range']:.3f}")


if __name__ == "__main__":
    main()
