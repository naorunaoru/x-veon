#!/usr/bin/env python3
"""Classify RAF files by high-frequency high-amplitude content.

Metric: 99th percentile of (Sobel gradient magnitude × pixel intensity).
Captures specular highlights, street lamps, headlights — bright areas with sharp edges.
"""

import json
import os
import sys
import glob
import numpy as np
import rawpy
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import ndimage

DCIM_ROOT = "/Volumes/4T/naoru/DCIM"
OUTPUT_JSON = "/Volumes/4T/xtrans-demosaic/datasets/raf_classification.json"


def process_file(path: str) -> dict | None:
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT,
                output_bps=16,
                no_auto_bright=True,
                no_auto_scale=True,
                gamma=(1, 1),
                output_color=rawpy.ColorSpace.raw,
                use_camera_wb=True,
                user_flip=0,
            )

        # Convert to float32 for computation
        img = rgb.astype(np.float32)

        # Luminance (simple average since we're in raw color space)
        lum = img.mean(axis=2)

        # Sobel gradient magnitude
        gx = ndimage.sobel(lum, axis=1)
        gy = ndimage.sobel(lum, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Metric: gradient magnitude × intensity, 99th percentile
        hfha = grad_mag * lum
        metric = float(np.percentile(hfha, 99))
        max_range = float(img.max())

        return {
            "file": os.path.basename(path),
            "path": path,
            "max_range": max_range,
            "hfha_metric": metric,
        }
    except Exception as e:
        print(f"ERROR {path}: {e}", file=sys.stderr)
        return None


def main():
    # Find all RAF files
    patterns = [os.path.join(DCIM_ROOT, "*_FUJI", "*.RAF")]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files.sort()

    print(f"Found {len(files)} RAF files")

    results = []
    workers = os.cpu_count() or 4

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_file, f): f for f in files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 50 == 0 or done == len(files):
                print(f"Progress: {done}/{len(files)}")
            result = fut.result()
            if result:
                results.append(result)

    # Sort by metric descending
    results.sort(key=lambda x: x["hfha_metric"], reverse=True)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} files classified.")
    print(f"Output: {OUTPUT_JSON}")
    print(f"Top 5 by HFHA metric:")
    for r in results[:5]:
        print(f"  {r['file']}: metric={r['hfha_metric']:.1f}, max={r['max_range']:.1f}")


if __name__ == "__main__":
    main()
