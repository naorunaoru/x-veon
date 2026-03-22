#!/usr/bin/env python3
"""
Classify raw images by high-frequency high-amplitude content.

Pipeline per image:
1. rawpy DHT demosaic, camera WB, linear (no gamma), no color matrix
2. Black-subtract, normalize by (white - black)
3. Compute luminance = 0.2126*R + 0.7152*G + 0.0722*B (linear)
4. Sobel gradient magnitude on luminance
5. Metric = 99th percentile of (gradient * luminance)  — "bright edges"
6. Also record max pixel value across channels

Output: JSON sorted by metric descending.
"""
import argparse
import os
import sys
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import rawpy


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


def process_raw(args):
    """Process one raw file, return dict with metrics or None on error."""
    raw_path, index, total = args

    stem = Path(raw_path).stem
    folder = Path(raw_path).parent.name

    try:
        raw = rawpy.imread(raw_path)

        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)
        scale = white - black
        if scale <= 0:
            return None

        # Max range: apply WB to per-channel raw maxes (unclipped)
        raw_image = raw.raw_image_visible.astype(np.float32)
        wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
        wb = wb / wb.min()
        colors = raw.raw_colors_visible
        channel_maxes = np.array([
            float(np.max(raw_image[colors == c])) if np.any(colors == c) else 0.0
            for c in range(3)
        ])
        max_val = float(np.max((channel_maxes - black) / scale * wb))

        # Pick demosaic algo: DHT for X-Trans (6x6), AHD for Bayer (2x2)
        is_xtrans = raw.raw_pattern.shape[0] == 6
        demosaic = rawpy.DemosaicAlgorithm.DHT if is_xtrans else rawpy.DemosaicAlgorithm.AHD

        rgb = raw.postprocess(
            demosaic_algorithm=demosaic,
            output_bps=16,
            no_auto_bright=True,
            no_auto_scale=True,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.raw,
            use_camera_wb=True,
            use_auto_wb=False,
            user_flip=0,
            half_size=True,
            highlight_mode=rawpy.HighlightMode.ReconstructDefault
        ).astype(np.float32)

        raw.close()

        # Normalize
        rgb = (rgb - black) / scale

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
            "path": raw_path,
            "folder": folder,
            "max_range": round(max_val, 4),
            "hf_ha_p99": round(metric, 6),
            "hf_ha_p999": round(metric_999, 6),
        }

    except Exception as e:
        print(f"  [{index}/{total}] {stem}: ERROR {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Classify raw images by high-frequency high-amplitude content")
    parser.add_argument("input_dir", help="Directory to scan for raw files")
    parser.add_argument("-o", "--output", help="Output JSON path (default: <input_dir>/hf_ha_ranking.json)")
    parser.add_argument("-e", "--ext", default="RAF",
                        help="Raw file extension to scan for (default: RAF)")
    parser.add_argument("-j", "--jobs", type=int, default=0,
                        help="Number of worker processes (default: all CPUs)")
    args = parser.parse_args()

    input_dir = args.input_dir
    ext = args.ext.upper().lstrip(".")
    output_json = args.output or os.path.join(input_dir, "hf_ha_ranking.json")

    # Find all raw files (skip ._ macOS resource forks)
    print(f"Scanning {input_dir} for .{ext} files...")
    raw_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if f.upper().endswith(f'.{ext}') and not f.startswith('._'):
                raw_files.append(os.path.join(root, f))

    raw_files.sort()
    total = len(raw_files)
    print(f"Found {total} .{ext} files")

    if total == 0:
        print("Nothing to do.")
        return

    # Build args
    work_args = [(path, i, total) for i, path in enumerate(raw_files)]

    # Process with multiprocessing
    ncpu = args.jobs if args.jobs > 0 else cpu_count()
    print(f"Processing with {ncpu} workers...")
    t0 = time.time()

    results = []
    with Pool(ncpu) as pool:
        for result in pool.imap_unordered(process_raw, work_args, chunksize=4):
            if result is not None:
                results.append(result)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(results)}/{total} images in {elapsed:.1f}s ({elapsed/total:.2f}s/img)")

    # Sort by metric descending
    results.sort(key=lambda x: x["hf_ha_p99"], reverse=True)

    # Save
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_json}")

    # Print top 20
    print("\nTop 20 high-frequency high-amplitude images:")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:3d}. {r['filename']:20s}  p99={r['hf_ha_p99']:.4f}  p999={r['hf_ha_p999']:.4f}  max={r['max_range']:.3f}")


if __name__ == "__main__":
    main()
