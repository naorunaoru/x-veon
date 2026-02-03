#!/usr/bin/env python3
"""
Process a single RAF through multiple checkpoints for comparison.

Usage:
    python3 compare_checkpoints.py --input test.RAF --checkpoints checkpoints_archive/
    python3 compare_checkpoints.py --input test.RAF --checkpoints v3.pt v4.pt
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import XTransUNet
from infer import load_raf, infer_tiled, find_pattern_shift


def main():
    parser = argparse.ArgumentParser(description='Compare checkpoints on a single RAF')
    parser.add_argument('--input', type=str, required=True, help='RAF file')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Checkpoint files or directory containing them')
    parser.add_argument('--output-dir', type=str, default='comparison')
    parser.add_argument('--patch-size', type=int, default=288)
    parser.add_argument('--overlap', type=int, default=48)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--preview-width', type=int, default=1200,
                        help='Width for preview JPEGs (0 = full size PNG only)')
    args = parser.parse_args()

    # Resolve checkpoints
    checkpoint_files = []
    for cp in args.checkpoints:
        p = Path(cp)
        if p.is_dir():
            checkpoint_files.extend(sorted(p.glob('*.pt')))
        elif p.is_file():
            checkpoint_files.append(p)
        else:
            print(f"WARNING: {cp} not found, skipping")
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        sys.exit(1)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load RAF once
    raf_path = args.input
    raf_stem = Path(raf_path).stem
    print(f"\nLoading: {raf_path}")
    cfa_srgb, pattern, meta = load_raf(raf_path)
    orig_h, orig_w = meta['original_shape']
    row_shift, col_shift = meta['shift']
    print(f"  Size: {orig_w}x{orig_h}, DR: {meta['dr_gain']}x, shift: ({row_shift},{col_shift})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing with {len(checkpoint_files)} checkpoint(s)...")
    
    for cp_path in checkpoint_files:
        cp_name = cp_path.stem
        print(f"\n{'='*60}")
        print(f"Checkpoint: {cp_name}")
        t0 = time.time()

        # Load model
        checkpoint = torch.load(str(cp_path), map_location=device, weights_only=False)
        model = XTransUNet(in_channels=4, out_channels=3)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            epoch = checkpoint.get('epoch', '?')
            psnr = checkpoint.get('best_val_psnr', '?')
            print(f"  Epoch {epoch}, val PSNR: {psnr}")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()

        # Infer
        rgb_float = infer_tiled(model, cfa_srgb, device,
                                patch_size=args.patch_size, overlap=args.overlap)
        
        # Remove alignment padding
        if row_shift != 0 or col_shift != 0:
            rgb_float = rgb_float[row_shift:row_shift+orig_h, col_shift:col_shift+orig_w, :]
        else:
            rgb_float = rgb_float[:orig_h, :orig_w, :]

        rgb = (rgb_float * 255).clip(0, 255).astype(np.uint8)
        
        elapsed = time.time() - t0

        # Save full-size PNG
        out_png = output_dir / f"{raf_stem}_{cp_name}.png"
        Image.fromarray(rgb).save(str(out_png), optimize=True)
        print(f"  Saved: {out_png} ({elapsed:.1f}s)")

        # Save preview JPEG
        if args.preview_width > 0:
            h, w = rgb.shape[:2]
            new_w = args.preview_width
            new_h = int(h * new_w / w)
            preview = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
            out_jpg = output_dir / f"{raf_stem}_{cp_name}_preview.jpg"
            preview.save(str(out_jpg), quality=90)
            print(f"  Preview: {out_jpg}")

        # Free model memory
        del model, checkpoint
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nAll done! Outputs in {output_dir}/")


if __name__ == '__main__':
    main()
