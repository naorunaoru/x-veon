#!/usr/bin/env python3
"""
Build training dataset from raw files.

Supports any raw format readable by rawpy (RAF, CR2, CR3, NEF, ARW, DNG, etc.).
Auto-detects sensor type (X-Trans vs Bayer) and selects appropriate demosaic algorithm.

Input modes:
  - JSON file: ranking/classification list (see load_raw_map for formats)
  - --scan-dir: scan a directory tree for raw files directly

DR-aware push (--dr-push):
  Reads Fuji DevelopmentDynamicRange from EXIF makernotes.
  DR400 images are underexposed by 2 stops → pushed +2 EV (×4).
  DR200 images are underexposed by 1 stop → pushed +1 EV (×2).
  Use --max-pushed-range to exclude images that still clip after push.

For each raw file:
1. Demosaic (DHT for X-Trans, AHD for Bayer)
2. Black-subtract, normalize by (white - black), NO CLIP
3. Apply DR push if enabled
4. Downscale 4x via area averaging
5. Save as float32 .npy
"""
import argparse
import os
import sys
import subprocess
import time
import json
import gc
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import rawpy


DOWNSAMPLE = 4

RAW_EXTENSIONS = {'.RAF', '.CR2', '.CR3', '.NEF', '.NRW', '.ARW', '.SRW',
                  '.RW2', '.ORF', '.PEF', '.IIQ'}

DR_GAIN = {400: 4.0, 200: 2.0, 100: 1.0}


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


def scan_dr(paths: list[str]) -> dict[str, int]:
    """Batch-query Fuji DevelopmentDynamicRange via exiftool. Returns {path: dr_value}."""
    if not paths:
        return {}
    result = subprocess.run(
        ['exiftool', '-json', '-DevelopmentDynamicRange'] + paths,
        capture_output=True, text=True, timeout=600,
    )
    dr_map = {}
    for entry in json.loads(result.stdout):
        dr_map[entry['SourceFile']] = entry.get('DevelopmentDynamicRange', 100)
    return dr_map


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


def process_raw(args):
    """Process a single raw file: demosaic, downsample, save."""
    raw_path, output_dir, stem, index, total, dr_gain, max_pushed_range = args

    output_path = os.path.join(output_dir, f"{stem}.npy")

    # Skip if already processed
    if os.path.exists(output_path):
        return f"  [{index}/{total}] {stem}: exists, skipping"

    try:
        raw = rawpy.imread(raw_path)

        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)

        # Auto-detect sensor type from CFA pattern
        raw_pat = raw.raw_pattern
        pat_h = raw_pat.shape[0]

        if pat_h >= 6:
            # X-Trans sensor
            demosaic_algo = rawpy.DemosaicAlgorithm.DHT
            sensor_type = "xtrans"
        else:
            # Bayer sensor
            demosaic_algo = rawpy.DemosaicAlgorithm.AHD
            sensor_type = "bayer"

        rgb_16 = raw.postprocess(
            demosaic_algorithm=demosaic_algo,
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

        # DR push: compensate for deliberate underexposure in DR200/400
        if dr_gain > 1.0:
            rgb_f *= dr_gain

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

        # Skip if pushed range exceeds threshold (highlights still clipped)
        if max_pushed_range is not None and downscaled.max() > max_pushed_range:
            dr_label = f" DR×{dr_gain:.0f}" if dr_gain > 1.0 else ""
            return f"  [{index}/{total}] {stem} ({sensor_type}{dr_label}): SKIPPED max_range={downscaled.max():.3f} > {max_pushed_range}"

        # Save
        np.save(output_path, downscaled)

        # Save metadata
        meta = {
            'source': raw_path,
            'sensor_type': sensor_type,
            'black_level': black,
            'white_level': white,
            'camera_wb': list(raw.camera_whitebalance[:3]),
            'original_size': [w, h],
            'downscaled_size': [new_w, new_h],
            'pattern': [[int(v) for v in row] for row in raw.raw_pattern],
            'range_min': float(downscaled.min()),
            'range_max': float(downscaled.max()),
            'dr_gain': dr_gain,
        }

        meta_path = os.path.join(output_dir, f"{stem}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        raw.close()

        dr_label = f" DR×{dr_gain:.0f}" if dr_gain > 1.0 else ""
        return f"  [{index}/{total}] {stem} ({sensor_type}{dr_label}): {new_w}x{new_h} range=[{downscaled.min():.3f}, {downscaled.max():.3f}]"

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
    parser.add_argument("--dr-push", action="store_true",
                        help="Apply DR-aware exposure push (DR400→×4, DR200→×2). Reads Fuji EXIF makernotes.")
    parser.add_argument("--dr-min", type=int, default=0,
                        help="Only include files with DR >= this value (e.g. 200 or 400). Requires --dr-push.")
    parser.add_argument("--max-pushed-range", type=float, default=None,
                        help="Exclude files whose pushed max_range would exceed this (e.g. 1.5). "
                             "Estimated as raw max_range × dr_gain. Requires --dr-push.")
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

    # DR scanning
    dr_per_file = {}  # stem -> dr_gain
    if args.dr_push:
        print("Scanning EXIF for DevelopmentDynamicRange...")
        path_to_stem = {v: k for k, v in raw_map.items()}
        dr_raw = scan_dr(list(raw_map.values()))
        for path, dr_val in dr_raw.items():
            stem = path_to_stem.get(path)
            if stem:
                dr_per_file[stem] = DR_GAIN.get(dr_val, 1.0)

        # Distribution
        dr_counts = {}
        for gain in dr_per_file.values():
            dr_counts[gain] = dr_counts.get(gain, 0) + 1
        for gain in sorted(dr_counts):
            label = {4.0: "DR400", 2.0: "DR200", 1.0: "DR100"}.get(gain, f"×{gain}")
            print(f"  {label}: {dr_counts[gain]} images")

        # Filter by --dr-min
        if args.dr_min > 0:
            min_gain = DR_GAIN.get(args.dr_min, 1.0)
            before = len(raw_map)
            raw_map = {k: v for k, v in raw_map.items()
                       if dr_per_file.get(k, 1.0) >= min_gain}
            print(f"  --dr-min {args.dr_min}: {before} → {len(raw_map)} files")

    # Create output dir
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Check how many already done
    existing = set(f.replace('.npy', '') for f in os.listdir(output_dir) if f.endswith('.npy') and not f.endswith('_lum.npy'))
    remaining = {k: v for k, v in raw_map.items() if k not in existing}
    print(f"Already processed: {len(existing)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return

    # Prepare work items
    work_args = [
        (path, output_dir, stem, i+1, len(remaining),
         dr_per_file.get(stem, 1.0), args.max_pushed_range)
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
