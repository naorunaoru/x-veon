#!/usr/bin/env python3
"""Backfill missing _meta.json files for .npy datasets.

Reads RAF headers to extract camera_wb and other metadata.
Does not re-demosaic — just reads the raw header.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import rawpy


RAF_BASE = "/Volumes/4T/naoru/DCIM"


def find_raf(stem: str, base_dir: str) -> str | None:
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for ext in ('.RAF', '.raf'):
            path = os.path.join(subdir_path, stem + ext)
            if os.path.exists(path):
                return path
    return None


def backfill(dataset_dir: str):
    npy_files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.endswith('.npy') and not f.endswith('_lum.npy') and not f.endswith('_meta.npy')
    ])

    missing = []
    for f in npy_files:
        stem = os.path.splitext(f)[0]
        meta_path = os.path.join(dataset_dir, f"{stem}_meta.json")
        if not os.path.exists(meta_path):
            missing.append(stem)

    print(f"Dataset: {dataset_dir}")
    print(f"  Total .npy: {len(npy_files)}, missing metadata: {len(missing)}")

    if not missing:
        print("  Nothing to backfill.")
        return

    filled = 0
    for i, stem in enumerate(missing):
        raf_path = find_raf(stem, RAF_BASE)
        if raf_path is None:
            print(f"  [{i+1}/{len(missing)}] {stem}: RAF not found, skipping")
            continue

        try:
            raw = rawpy.imread(raf_path)
            npy_path = os.path.join(dataset_dir, f"{stem}.npy")
            arr = np.load(npy_path, mmap_mode='r')

            meta = {
                'source': raf_path,
                'black_level': float(raw.black_level_per_channel[0]),
                'white_level': float(raw.white_level),
                'camera_wb': [float(x) for x in raw.camera_whitebalance[:3]],
                'original_size': [raw.sizes.width, raw.sizes.height],
                'downscaled_size': [arr.shape[1], arr.shape[0]],
                'pattern': [list(int(x) for x in row) for row in raw.raw_pattern],
                'range_min': float(arr.min()),
                'range_max': float(arr.max()),
            }
            raw.close()

            meta_path = os.path.join(dataset_dir, f"{stem}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

            filled += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(missing)}] {filled} filled...")

        except Exception as e:
            print(f"  [{i+1}/{len(missing)}] {stem}: ERROR - {e}")

    print(f"  Done: {filled}/{len(missing)} metadata files created")


if __name__ == '__main__':
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/4T/xtrans-demosaic/datasets/top2000_dataset"
    backfill(dataset_dir)
