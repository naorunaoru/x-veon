#!/usr/bin/env python3
"""
Preview dataset samples with augmentations applied.

Usage:
    python preview_dataset.py --data-dir /path/to/npy --n 8 --bright-spot-prob 1.0 -o preview.png
"""

import argparse
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import LinearDataset


def gamma_correct(rgb: torch.Tensor) -> np.ndarray:
    """(3, H, W) linear → (H, W, 3) uint8 sRGB."""
    arr = rgb.clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr ** (1 / 2.2) * 255).astype(np.uint8)


def cfa_false_color(cfa_img: torch.Tensor, cfa_mask: torch.Tensor) -> np.ndarray:
    """Colorize CFA mosaic by channel: R sites red, G green, B blue. Returns (H, W, 3) uint8."""
    H, W = cfa_mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    vals = cfa_img[0].clamp(0, 1).numpy()
    vals_u8 = (vals ** (1 / 2.2) * 255).astype(np.uint8)
    for ch in range(3):
        mask = (cfa_mask == ch).numpy()
        out[mask, ch] = vals_u8[mask]
    return out



def main():
    parser = argparse.ArgumentParser(description="Preview dataset samples")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--cfa-type", type=str, default="xtrans")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("-n", "--n-samples", type=int, default=8)
    parser.add_argument("-o", "--output", type=str, default="preview.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply-wb", action="store_true")
    # Augmentation params
    parser.add_argument("--bright-spot-prob", type=float, default=0.0)
    parser.add_argument("--bright-spot-intensity-max", type=float, default=5.0)
    parser.add_argument("--bright-spot-sigma-max", type=float, default=20.0)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.0)
    args = parser.parse_args()

    ds = LinearDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        augment=True,
        noise_sigma=(args.noise_min, args.noise_max),
        apply_wb=args.apply_wb,
        cfa_type=args.cfa_type,
        bright_spot_prob=args.bright_spot_prob,
        bright_spot_intensity=(1.5, args.bright_spot_intensity_max),
        bright_spot_sigma=(2.0, args.bright_spot_sigma_max),
    )

    print(f"Dataset: {len(ds)} samples from {len(ds.data_files)} images")
    print(f"CFA: {args.cfa_type}, patch: {args.patch_size}px")

    # Column labels
    col_labels = ["Ground Truth", "CFA Mosaic"]
    n_cols = len(col_labels)
    ps = args.patch_size
    label_h = 20
    pad = 2

    grid_w = n_cols * ps + (n_cols - 1) * pad
    grid_h = args.n_samples * ps + (args.n_samples - 1) * pad + label_h
    grid = Image.new('RGB', (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    # Draw column labels
    for j, label in enumerate(col_labels):
        x = j * (ps + pad) + ps // 2
        draw.text((x, 2), label, fill=(200, 200, 200), anchor="mt")

    rng = random.Random(args.seed)
    indices = [rng.randint(0, len(ds) - 1) for _ in range(args.n_samples)]

    for i, idx in enumerate(indices):
        input_tensor, ref = ds[idx]

        cfa_img = input_tensor[0:1]  # (1, H, W)

        gt_u8 = gamma_correct(ref)
        cfa_u8 = cfa_false_color(cfa_img, ds.cfa)

        y = i * (ps + pad) + label_h
        for j, arr in enumerate([gt_u8, cfa_u8]):
            x = j * (ps + pad)
            grid.paste(Image.fromarray(arr), (x, y))

    grid.save(args.output)
    print(f"Saved {args.output} ({grid_w}x{grid_h})")


if __name__ == "__main__":
    main()
