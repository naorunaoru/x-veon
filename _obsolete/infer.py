#!/usr/bin/env python3
"""
X-Trans demosaicing inference pipeline.

Takes RAF files, runs them through the trained U-Net model,
and outputs full-resolution demosaiced images.

Usage:
    python3 infer.py --checkpoint checkpoints_v3/best.pt --input photo.RAF --output result.png
    python3 infer.py --checkpoint checkpoints_v3/best.pt --input-dir raw_photos/ --output-dir results/
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rawpy
import torch
from PIL import Image

from model import XTransUNet
from xtrans_pattern import make_cfa_mask, make_channel_masks, XTRANS_PATTERN


def get_dr_gain(filepath: str) -> float:
    """
    Extract Dynamic Range gain from RAF metadata using exiftool.
    
    DR100 = 1.0 (no gain)
    DR200 = 2.0 (1 stop underexposed)
    DR400 = 4.0 (2 stops underexposed)
    """
    try:
        result = subprocess.run(
            ['exiftool', '-DevelopmentDynamicRange', '-s3', str(filepath)],
            capture_output=True, text=True, timeout=5
        )
        dr_value = int(result.stdout.strip())
        gain = dr_value / 100.0
        return gain
    except Exception:
        # Try homebrew path
        try:
            result = subprocess.run(
                ['/opt/homebrew/bin/exiftool', '-DevelopmentDynamicRange', '-s3', str(filepath)],
                capture_output=True, text=True, timeout=5
            )
            dr_value = int(result.stdout.strip())
            return dr_value / 100.0
        except Exception:
            print("  WARNING: Could not read DR metadata, assuming DR100")
            return 1.0


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma curve to linear data."""
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(np.clip(x, 0.0031308, None), 1.0/2.4) - 0.055
    )


def find_pattern_shift(file_pattern: np.ndarray) -> tuple[int, int]:
    """
    Find the row and column shift needed to align a file's CFA pattern
    with the training pattern (XTRANS_PATTERN).
    
    The X-Trans pattern is always the same 6x6 tile, but different cameras
    or orientations may start at a different offset within the tile.
    
    Returns:
        (row_shift, col_shift): how many rows/cols to pad the raw image
        so its CFA aligns with XTRANS_PATTERN.
    """
    if np.array_equal(file_pattern, XTRANS_PATTERN):
        return (0, 0)
    
    # Try all possible shifts within the 6x6 tile
    for row_shift in range(6):
        for col_shift in range(6):
            shifted = np.roll(np.roll(XTRANS_PATTERN, -row_shift, axis=0), -col_shift, axis=1)
            if np.array_equal(file_pattern, shifted):
                return (row_shift, col_shift)
    
    # No match found — warn and return no shift
    print(f"  WARNING: Could not find pattern shift! File pattern:")
    print(f"  {file_pattern}")
    return (0, 0)


def load_raf(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a Fujifilm RAF file and prepare CFA data with proper exposure.
    
    Returns:
        cfa_srgb: (H, W) float32 array in [0, 1] — CFA values in sRGB-like space,
                  aligned to XTRANS_PATTERN (may be padded)
        pattern: (6, 6) int array — CFA pattern from the file (original)
        meta: dict with metadata including 'shift' and 'original_shape'
    """
    # Get DR gain first
    dr_gain = get_dr_gain(path)
    
    with rawpy.imread(path) as raw:
        pattern = raw.raw_pattern.copy()
        
        # Raw image (linear, 14-bit typically)
        raw_image = raw.raw_image_visible.copy().astype(np.float32)
        
        # Black/white levels
        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)
        
        # Camera white balance (normalize to green=1)
        cam_wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
        cam_wb = cam_wb / cam_wb[1]  # Normalize to G=1
        
        meta = {
            'black': black,
            'white': white,
            'camera_wb': cam_wb.tolist(),
            'dr_gain': dr_gain,
            'pattern': pattern.tolist(),
        }
    
    # Find CFA pattern alignment shift
    row_shift, col_shift = find_pattern_shift(pattern)
    meta['shift'] = (row_shift, col_shift)
    meta['original_shape'] = raw_image.shape
    
    if row_shift != 0 or col_shift != 0:
        print(f"  CFA pattern shift: ({row_shift}, {col_shift}) — padding to align with training pattern")
        # Pad the raw image so CFA aligns with XTRANS_PATTERN
        # Pad at the top/left with reflected pixels
        raw_image = np.pad(raw_image, ((row_shift, 0), (col_shift, 0)), mode='reflect')
    
    h, w = raw_image.shape
    
    # Build CFA mask using the TRAINING pattern (now aligned)
    cfa_mask = np.tile(XTRANS_PATTERN, ((h+5)//6, (w+5)//6))[:h, :w]
    
    # 1. Subtract black level and normalize to [0, 1] linear
    raw_linear = (raw_image - black) / (white - black)
    raw_linear = np.clip(raw_linear, 0.0, 1.0)
    
    # 2. Apply DR gain (compensate for intentional underexposure)
    raw_linear = raw_linear * dr_gain
    
    # 3. Apply white balance per-pixel based on CFA position
    for ch in range(3):  # R=0, G=1, B=2
        mask = (cfa_mask == ch)
        raw_linear[mask] *= cam_wb[ch]
    
    # Clip after WB (some channels may exceed 1.0)
    raw_linear = np.clip(raw_linear, 0.0, 1.0)
    
    # 4. Apply sRGB gamma (model was trained on sRGB JPEG data)
    cfa_srgb = linear_to_srgb(raw_linear).astype(np.float32)
    
    return cfa_srgb, pattern, meta


def make_hann_window(h: int, w: int) -> torch.Tensor:
    """Create a 2D Hann window for smooth patch blending."""
    win_h = torch.hann_window(h, periodic=False)
    win_w = torch.hann_window(w, periodic=False)
    return win_h.unsqueeze(1) * win_w.unsqueeze(0)


def infer_tiled(
    model: XTransUNet,
    cfa_srgb: np.ndarray,
    device: torch.device,
    patch_size: int = 288,
    overlap: int = 96,
) -> np.ndarray:
    """
    Run inference on a full-resolution image using overlapping tiles.
    
    Returns:
        (H, W, 3) float32 RGB image in [0, 1] (sRGB gamma space)
    """
    h, w = cfa_srgb.shape
    stride = patch_size - overlap
    
    # Pad image to fit tiles exactly
    pad_h = (stride - (h - patch_size) % stride) % stride if h > patch_size else patch_size - h
    pad_w = (stride - (w - patch_size) % stride) % stride if w > patch_size else patch_size - w
    
    cfa_padded = np.pad(cfa_srgb, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = cfa_padded.shape
    
    # Output accumulator and weight map
    output = torch.zeros(3, ph, pw, device='cpu')
    weights = torch.zeros(1, ph, pw, device='cpu')
    
    window = make_hann_window(patch_size, patch_size)
    
    # Patch positions
    positions = []
    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            positions.append((y, x))
    
    # CFA position masks (same for all patches since pattern tiles with XTRANS_PATTERN)
    masks = make_channel_masks(patch_size, patch_size).to(device)
    
    print(f"  Image: {h}x{w} -> padded {ph}x{pw}")
    print(f"  Patches: {len(positions)} ({patch_size}x{patch_size}, stride {stride})")
    
    model.eval()
    with torch.no_grad():
        for i, (y, x) in enumerate(positions):
            cfa_patch = cfa_padded[y:y+patch_size, x:x+patch_size]
            cfa_tensor = torch.from_numpy(cfa_patch).float().unsqueeze(0).to(device)
            
            inp = torch.cat([cfa_tensor, masks], dim=0).unsqueeze(0)
            
            pred = model(inp).squeeze(0).cpu().clamp(0, 1)
            
            output[:, y:y+patch_size, x:x+patch_size] += pred * window
            weights[:, y:y+patch_size, x:x+patch_size] += window
            
            if (i + 1) % 100 == 0 or i == len(positions) - 1:
                print(f"  Patch {i+1}/{len(positions)}")
    
    weights = weights.clamp(min=1e-8)
    output = output / weights
    output = output[:, :h, :w]
    
    return output.permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(description='X-Trans demosaicing inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, help='Single RAF file')
    parser.add_argument('--input-dir', type=str, help='Directory of RAF files')
    parser.add_argument('--output', type=str, help='Output path (single file)')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--patch-size', type=int, default=288)
    parser.add_argument('--overlap', type=int, default=96)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
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
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model = XTransUNet(in_channels=4, out_channels=3)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch', '?')
        psnr = checkpoint.get('best_val_psnr', '?')
        print(f"  Epoch {epoch}, PSNR: {psnr}")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    
    # Gather inputs
    if args.input:
        input_files = [Path(args.input)]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        input_files = sorted(
            list(input_dir.glob('*.RAF')) + list(input_dir.glob('*.raf')) +
            list(input_dir.glob('*.DNG')) + list(input_dir.glob('*.dng'))
        )
    else:
        print("Error: specify --input or --input-dir")
        sys.exit(1)
    
    if not input_files:
        print("No input files found!")
        sys.exit(1)
    
    print(f"Processing {len(input_files)} file(s)...")
    
    output_dir = Path(args.output_dir) if not args.output else Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filepath in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {filepath.name}")
        t0 = time.time()
        
        # Load and prepare RAF
        cfa_srgb, pattern, meta = load_raf(str(filepath))
        orig_h, orig_w = meta['original_shape']
        row_shift, col_shift = meta['shift']
        print(f"  Raw size: {orig_w}x{orig_h}")
        print(f"  DR gain: {meta['dr_gain']}x  WB: R={meta['camera_wb'][0]:.2f} G={meta['camera_wb'][1]:.2f} B={meta['camera_wb'][2]:.2f}")
        print(f"  CFA range after processing: [{cfa_srgb.min():.3f}, {cfa_srgb.max():.3f}]  mean: {cfa_srgb.mean():.3f}")
        
        # Check pattern
        if pattern.shape != (6, 6):
            print(f"  WARNING: Unexpected CFA pattern shape {pattern.shape}")
        
        # Run inference on the (possibly padded) image
        rgb_float = infer_tiled(
            model, cfa_srgb, device,
            patch_size=args.patch_size,
            overlap=args.overlap,
        )
        
        # Remove the alignment padding to get back to original dimensions
        if row_shift != 0 or col_shift != 0:
            rgb_float = rgb_float[row_shift:row_shift+orig_h, col_shift:col_shift+orig_w, :]
        else:
            rgb_float = rgb_float[:orig_h, :orig_w, :]
        
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        
        # Convert to uint8
        rgb = (rgb_float * 255).clip(0, 255).astype(np.uint8)
        
        # Save
        if args.output and len(input_files) == 1:
            out_path = Path(args.output)
        else:
            out_path = output_dir / f"{filepath.stem}_demosaic.png"
        
        Image.fromarray(rgb).save(str(out_path), optimize=True)
        print(f"  Saved: {out_path} ({rgb.shape[1]}x{rgb.shape[0]})")
    
    print(f"\nAll done!")


if __name__ == '__main__':
    main()
