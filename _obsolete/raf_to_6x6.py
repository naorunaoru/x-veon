#!/usr/bin/env python3
"""
Produce 1/6 scale true RGB images from RAF files by averaging each 6x6 CFA tile.

Each 6x6 X-Trans tile has 8R, 20G, 8B photosites. Averaging them gives one
pixel with true sensor RGB values — no interpolation, no artifacts.

Outputs both linear EXR (full dynamic range) and tonemapped PNG for viewing.
"""
import sys
import subprocess
import numpy as np
import rawpy
from pathlib import Path
from PIL import Image


XTRANS_PATTERN = np.array([
    [0, 2, 1, 2, 0, 1],
    [1, 1, 0, 1, 1, 2],
    [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1],
    [1, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 2],
], dtype=np.int32)


def find_pattern_shift(file_pattern):
    """Find row/col shift to align file's CFA with XTRANS_PATTERN."""
    if np.array_equal(file_pattern, XTRANS_PATTERN):
        return (0, 0)
    for rs in range(6):
        for cs in range(6):
            shifted = np.roll(np.roll(XTRANS_PATTERN, -rs, axis=0), -cs, axis=1)
            if np.array_equal(file_pattern, shifted):
                return (rs, cs)
    print("  WARNING: Could not find pattern shift!")
    return (0, 0)


def get_dr_gain(filepath):
    """Extract DR gain from RAF metadata."""
    for exiftool in ['exiftool', '/opt/homebrew/bin/exiftool']:
        try:
            result = subprocess.run(
                [exiftool, '-DevelopmentDynamicRange', '-s3', str(filepath)],
                capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip()) / 100.0
        except Exception:
            continue
    return 1.0


def linear_to_srgb(x):
    """Apply sRGB gamma for display."""
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(np.clip(x, 0.0031308, None), 1.0/2.4) - 0.055
    )


def raf_to_6x6(raf_path, output_dir):
    """Convert a RAF file to a 1/6 scale true RGB image."""
    raf_path = Path(raf_path)
    stem = raf_path.stem
    
    dr_gain = get_dr_gain(raf_path)
    
    with rawpy.imread(str(raf_path)) as raw:
        pattern = raw.raw_pattern.copy()
        raw_image = raw.raw_image_visible.copy().astype(np.float32)
        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)
        cam_wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
        cam_wb = cam_wb / cam_wb[1]
    
    # Align CFA pattern
    row_shift, col_shift = find_pattern_shift(pattern)
    if row_shift != 0 or col_shift != 0:
        raw_image = np.pad(raw_image, ((row_shift, 0), (col_shift, 0)), mode='reflect')
    
    h, w = raw_image.shape
    
    # Normalize to linear [0, 1]
    raw_linear = (raw_image - black) / (white - black)
    raw_linear = np.clip(raw_linear, 0.0, None)  # Don't clip top — preserve full DR
    
    # Apply DR gain
    raw_linear = raw_linear * dr_gain
    
    # Trim to multiple of 6
    h6 = (h // 6) * 6
    w6 = (w // 6) * 6
    raw_linear = raw_linear[:h6, :w6]
    
    # Build aligned CFA mask
    cfa_mask = np.tile(XTRANS_PATTERN, (h6 // 6, w6 // 6))
    
    # Reshape into 6x6 blocks
    blocks = raw_linear.reshape(h6 // 6, 6, w6 // 6, 6)
    mask_blocks = cfa_mask.reshape(h6 // 6, 6, w6 // 6, 6)
    
    # Average each channel within each 6x6 block
    out_h, out_w = h6 // 6, w6 // 6
    rgb_linear = np.zeros((out_h, out_w, 3), dtype=np.float32)
    
    for ch in range(3):
        ch_mask = (mask_blocks == ch)  # (out_h, 6, out_w, 6)
        ch_values = blocks * ch_mask
        ch_sum = ch_values.sum(axis=(1, 3))
        ch_count = ch_mask.sum(axis=(1, 3)).astype(np.float32)
        ch_count = np.maximum(ch_count, 1)  # Avoid div by zero
        rgb_linear[:, :, ch] = ch_sum / ch_count
    
    # Apply white balance
    for ch in range(3):
        rgb_linear[:, :, ch] *= cam_wb[ch]
    
    print(f"  {stem}: {w}x{h} -> {out_w}x{out_h}")
    print(f"    DR gain: {dr_gain}x, WB: R={cam_wb[0]:.2f} G={cam_wb[1]:.2f} B={cam_wb[2]:.2f}")
    print(f"    Linear range: [{rgb_linear.min():.4f}, {rgb_linear.max():.4f}]")
    print(f"    Per-channel mean: R={rgb_linear[:,:,0].mean():.4f} G={rgb_linear[:,:,1].mean():.4f} B={rgb_linear[:,:,2].mean():.4f}")
    
    # Save linear float32 numpy (preserves full dynamic range)
    npy_path = Path(output_dir) / f"{stem}_linear_6x6.npy"
    np.save(str(npy_path), rgb_linear)
    print(f"    Saved: {npy_path} ({rgb_linear.nbytes / 1024:.0f} KB)")
    
    # Save sRGB tonemapped PNG for viewing
    # Simple tonemap: clip to [0, 1] then apply sRGB gamma
    rgb_clipped = np.clip(rgb_linear, 0.0, 1.0)
    rgb_srgb = linear_to_srgb(rgb_clipped)
    png_data = (rgb_srgb * 255).clip(0, 255).astype(np.uint8)
    png_path = Path(output_dir) / f"{stem}_srgb_6x6.png"
    Image.fromarray(png_data).save(str(png_path))
    print(f"    Saved: {png_path}")
    
    # Also save a simple exposure-boosted version for dark images
    rgb_boosted = np.clip(rgb_linear * 1.5, 0.0, 1.0)
    rgb_boosted_srgb = linear_to_srgb(rgb_boosted)
    boosted_data = (rgb_boosted_srgb * 255).clip(0, 255).astype(np.uint8)
    boosted_path = Path(output_dir) / f"{stem}_srgb_bright_6x6.png"
    Image.fromarray(boosted_data).save(str(boosted_path))
    
    return rgb_linear


if __name__ == '__main__':
    import os
    
    raf_dir = sys.argv[1] if len(sys.argv) > 1 else 'test_rafs'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output_6x6'
    
    os.makedirs(output_dir, exist_ok=True)
    
    raf_files = sorted(Path(raf_dir).glob('*.RAF')) + sorted(Path(raf_dir).glob('*.raf'))
    
    if not raf_files:
        print(f"No RAF files found in {raf_dir}")
        sys.exit(1)
    
    print(f"Processing {len(raf_files)} RAF files -> {output_dir}/")
    
    for raf in raf_files:
        raf_to_6x6(str(raf), output_dir)
    
    print(f"\nDone!")
