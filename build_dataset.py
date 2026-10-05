#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
"""
Build training dataset from raw files.

Supports any raw format readable by rawpy (RAF, CR2, CR3, NEF, ARW, DNG, etc.).
Auto-detects sensor type (X-Trans vs Bayer) and selects appropriate demosaic algorithm.

Input modes:
  - JSON file: ranking/classification list (see load_raw_map for formats)
  - --scan-dir: scan a directory tree for raw files directly

For each raw file:
1. Demosaic with LibRaw (Markesteijn three-pass for X-Trans, AHD for Bayer)
2. Scale to 16 bits with the sensor's white level at 65535
3. Area-average 2x downscale
4. Round to uint16, save as .npy

The destination gets a build_info.json (see dataset_record.py). A directory that holds
.npy files from another revision, other options, or an older builder is refused.
"""
import argparse
import os
import sys
import time
import json
from multiprocessing import Pool, cpu_count
from typing import Any

import numpy as np
import rawpy

from dataset_record import claim_destination, new_record

DOWNSCALE = 2


RAW_EXTENSIONS = {'.RAF', '.CR2', '.CR3', '.NEF', '.NRW', '.ARW', '.SRW',
                  '.RW2', '.ORF', '.PEF', '.IIQ'}


def scan_dir(scan_path: str) -> dict[str, str]:
    """Scan a directory tree for raw files. Returns {stem: raw_path}."""
    raw_map = {}
    for root, _, files in os.walk(scan_path):
        for filename in sorted(files):
            if os.path.splitext(filename)[1].upper() not in RAW_EXTENSIONS:
                continue
            stem = os.path.splitext(filename)[0]
            raw_map[stem] = os.path.join(root, filename)
    return raw_map


def load_raw_map(json_path: str, raw_base: str | None = None, top_n: int | None = None) -> dict[str, str]:
    """Load a JSON ranking file and return {stem: raw_path} dict."""
    with open(json_path) as f:
        data = json.load(f)

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        raw_map = {}
        for entry in data:
            path = entry["path"]
            if "filename" in entry:
                stem = entry["filename"]
            else:
                stem = os.path.splitext(entry["file"])[0]
            raw_map[stem] = path
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        # Legacy: plain list of stems, needs raw_base
        if not raw_base:
            sys.exit("Legacy JSON (plain list of IDs) requires --raw-base")
        image_ids = set(data)
        raw_map = {}
        for subdir in os.listdir(raw_base):
            subdir_path = os.path.join(raw_base, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for filename in os.listdir(subdir_path):
                if os.path.splitext(filename)[1].upper() not in RAW_EXTENSIONS:
                    continue
                stem = os.path.splitext(filename)[0]
                if stem in image_ids:
                    raw_map[stem] = os.path.join(subdir_path, filename)
    else:
        sys.exit(f"Unrecognized JSON format in {json_path}")

    if top_n and top_n < len(raw_map):
        # Preserve order from the JSON (assumed pre-sorted by quality)
        raw_map = dict(list(raw_map.items())[:top_n])

    return raw_map


def demosaic_linear(raw: Any) -> tuple[np.ndarray, str]:
    """LibRaw's demosaic of an open rawpy image as (H, W, 3) uint16, and the sensor type.

    65535 is the sensor's white level in every image. adjust_maximum_thr=0 stops LibRaw
    from moving the white level down to the image's own maximum, which it otherwise does
    when that maximum lies between 75% and 100% of the sensor's.
    """
    pattern = raw.raw_pattern
    assert pattern is not None
    if pattern.shape[0] >= 6:
        # LibRaw runs Markesteijn (three passes) on every X-Trans file, whatever is requested.
        algorithm, sensor_type = rawpy.DemosaicAlgorithm.DHT, "xtrans"
    else:
        algorithm, sensor_type = rawpy.DemosaicAlgorithm.AHD, "bayer"
    rgb = raw.postprocess(
        demosaic_algorithm=algorithm,
        output_bps=16,
        no_auto_bright=True,
        adjust_maximum_thr=0.0,
        gamma=(1, 1),  # Linear
        output_color=rawpy.ColorSpace.raw,  # No color matrix
        use_camera_wb=False,
        use_auto_wb=False,
        user_wb=[1, 1, 1, 1],  # Unity WB
        highlight_mode=rawpy.HighlightMode.Ignore,  # Leave highlights unclipped (no reconstruction)
        half_size=False,  # Full resolution demosaic
        user_flip=0,  # No EXIF rotation - keep raw sensor orientation
    )
    return np.asarray(rgb, dtype=np.uint16), sensor_type


def process_raw(args):
    """Process a single raw file: demosaic, downsample, save."""
    raw_path, output_dir, stem, index, total = args

    output_path = os.path.join(output_dir, f"{stem}.npy")

    # Skip if already processed
    if os.path.exists(output_path):
        return f"  [{index}/{total}] {stem}: exists, skipping"

    try:
        raw = rawpy.imread(raw_path)
        rgb_16, sensor_type = demosaic_linear(raw)

        # Area-average 2x downscale, round back to uint16
        ds = DOWNSCALE
        h_crop = rgb_16.shape[0] // ds * ds
        w_crop = rgb_16.shape[1] // ds * ds
        rgb_ds = (rgb_16[:h_crop, :w_crop]
                  .reshape(h_crop // ds, ds, w_crop // ds, ds, 3)
                  .mean(axis=(1, 3), dtype=np.float32))
        del rgb_16
        rgb_u16 = np.rint(rgb_ds).astype(np.uint16)
        del rgb_ds

        h, w = rgb_u16.shape[:2]

        # Save
        np.save(output_path, rgb_u16)

        # Save metadata
        meta = {
            'source': raw_path,
            'sensor_type': sensor_type,
            'camera_wb': list(raw.camera_whitebalance[:3]),
            'original_size': [w * 2, h * 2],
            'downscaled_size': [w, h],
            'pattern': [[int(v) for v in row] for row in np.asarray(raw.raw_pattern)],
            'range_min': int(np.min(rgb_u16)),
            'range_max': int(np.max(rgb_u16)),
        }

        meta_path = os.path.join(output_dir, f"{stem}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        raw.close()

        return f"  [{index}/{total}] {stem} ({sensor_type}): {w}x{h} range=[{int(np.min(rgb_u16))}, {int(np.max(rgb_u16))}]"

    except Exception as e:
        return f"  [{index}/{total}] {stem}: ERROR - {e}"


def main():
    parser = argparse.ArgumentParser(description="Build demosaic training dataset from raw files")
    parser.add_argument("json_file", nargs='?', default=None,
                        help="Ranking JSON (hf_ha_ranking.json, or any [{path, filename}, ...] list)")
    parser.add_argument("--scan-dir", default=None,
                        help="Scan a directory tree for raw files (alternative to JSON input)")
    parser.add_argument("-o", "--output", required=True, help="Output directory for .npy files")
    parser.add_argument("-n", "--top-n", type=int, default=None, help="Only process the top N entries (default: all)")
    parser.add_argument("--raw-base", default=None, help="Base DCIM directory (only needed for legacy plain-list JSON)")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    if not args.json_file and not args.scan_dir:
        parser.error("Either json_file or --scan-dir is required")

    # Load raw file map
    if args.scan_dir:
        raw_map = scan_dir(args.scan_dir)
        print(f"Scanned {len(raw_map)} raw files from {args.scan_dir}")
        if args.top_n and args.top_n < len(raw_map):
            raw_map = dict(list(raw_map.items())[:args.top_n])
    else:
        raw_map = load_raw_map(args.json_file, raw_base=args.raw_base, top_n=args.top_n)
        print(f"Loaded {len(raw_map)} raw files from {args.json_file}")

    # Verify paths exist
    missing = {k: v for k, v in raw_map.items() if not os.path.exists(v)}
    if missing:
        print(f"WARNING: {len(missing)} raw files not found on disk (skipping)")
        for stem in list(missing)[:5]:
            print(f"  {missing[stem]}")
        raw_map = {k: v for k, v in raw_map.items() if k not in missing}

    # Create output dir
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    claim_destination(output_dir, new_record(
        {"source": args.scan_dir or args.json_file, "top_n": args.top_n, "raw_base": args.raw_base,
         "downscale": DOWNSCALE, "adjust_maximum_thr": 0.0},
        os.path.dirname(os.path.abspath(__file__)),
    ))

    # Check how many already done
    existing = set(f.replace('.npy', '') for f in os.listdir(output_dir) if f.endswith('.npy') and not f.endswith('_lum.npy'))
    remaining = {k: v for k, v in raw_map.items() if k not in existing}
    print(f"Already processed: {len(existing)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return

    # Prepare work items
    work_args = [
        (path, output_dir, stem, i+1, len(remaining))
        for i, (stem, path) in enumerate(remaining.items())
    ]

    # Process with limited parallelism (memory bounded)
    n_workers = min(args.workers, cpu_count())
    print(f"Processing with {n_workers} workers...")
    t0 = time.time()

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(process_raw, work_args):
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
