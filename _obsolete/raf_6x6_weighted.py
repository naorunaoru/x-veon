#!/usr/bin/env python3
"""
Test weighted 6x6 averaging — give higher weight to center photosites
to reduce color fringing at edges.
"""
import numpy as np
import rawpy
import subprocess
from PIL import Image

XTRANS_PATTERN = np.array([
    [0, 2, 1, 2, 0, 1],
    [1, 1, 0, 1, 1, 2],
    [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1],
    [1, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 2],
], dtype=np.int32)

def linear_to_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(np.clip(x, 0.0031308, None), 1.0/2.4) - 0.055)

def get_dr_gain(filepath):
    for exiftool in ['exiftool', '/opt/homebrew/bin/exiftool']:
        try:
            result = subprocess.run([exiftool, '-DevelopmentDynamicRange', '-s3', str(filepath)],
                                    capture_output=True, text=True, timeout=5)
            return int(result.stdout.strip()) / 100.0
        except Exception:
            continue
    return 1.0

# Load RAF
raw = rawpy.imread("test_rafs/DSCF3561.RAF")
vis = raw.raw_image_visible.copy().astype(np.float32)
pattern = raw.raw_pattern.copy()
black = float(raw.black_level_per_channel[0])
white = float(raw.white_level)
cam_wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
cam_wb = cam_wb / cam_wb[1]
dr_gain = get_dr_gain("test_rafs/DSCF3561.RAF")

# Normalize
raw_linear = (vis - black) / (white - black)
raw_linear = np.clip(raw_linear, 0.0, None)
raw_linear *= dr_gain

h, w = raw_linear.shape
h6 = (h // 6) * 6
w6 = (w // 6) * 6
raw_linear = raw_linear[:h6, :w6]

# Build 2D Gaussian-like weight map for 6x6 tile (center-biased)
# Distance from center (2.5, 2.5)
y_coords, x_coords = np.mgrid[0:6, 0:6]
dist = np.sqrt((y_coords - 2.5)**2 + (x_coords - 2.5)**2)
sigma = 1.5
spatial_weight = np.exp(-dist**2 / (2 * sigma**2))

print("Spatial weight map:")
print(np.round(spatial_weight, 2))

# Show which photosites get what weight per channel
for ch, name in enumerate("RGB"):
    mask = (XTRANS_PATTERN == ch)
    weighted = spatial_weight * mask
    print(f"\n{name} channel weighted positions (sum={weighted.sum():.2f}):")
    print(np.round(weighted, 2))

# Reshape into blocks
out_h, out_w = h6 // 6, w6 // 6
blocks = raw_linear.reshape(out_h, 6, out_w, 6)

# Tile weights
weight_tile = spatial_weight  # (6, 6)

rgb_weighted = np.zeros((out_h, out_w, 3), dtype=np.float32)
for ch in range(3):
    ch_mask = (XTRANS_PATTERN == ch).astype(np.float32)  # (6, 6)
    ch_weights = ch_mask * weight_tile  # (6, 6)
    ch_weight_sum = ch_weights.sum()
    
    # Apply weights: blocks is (out_h, 6, out_w, 6)
    # ch_weights is (6, 6) -> broadcast as (1, 6, 1, 6)
    weighted_sum = (blocks * ch_weights[np.newaxis, :, np.newaxis, :]).sum(axis=(1, 3))
    rgb_weighted[:, :, ch] = weighted_sum / ch_weight_sum

# Apply WB
for ch in range(3):
    rgb_weighted[:, :, ch] *= cam_wb[ch]

# Also do uniform average for comparison
rgb_uniform = np.zeros((out_h, out_w, 3), dtype=np.float32)
mask_blocks = np.tile(XTRANS_PATTERN, (out_h, out_w)).reshape(out_h, 6, out_w, 6)
for ch in range(3):
    ch_mask = (mask_blocks == ch)
    ch_sum = (blocks * ch_mask).sum(axis=(1, 3))
    ch_count = ch_mask.sum(axis=(1, 3)).astype(np.float32)
    rgb_uniform[:, :, ch] = ch_sum / np.maximum(ch_count, 1)
for ch in range(3):
    rgb_uniform[:, :, ch] *= cam_wb[ch]

# Save both
for name, data in [("weighted", rgb_weighted), ("uniform", rgb_uniform)]:
    clipped = np.clip(data, 0.0, 1.0)
    srgb = linear_to_srgb(clipped)
    out = (srgb * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(f"output_6x6/DSCF3561_{name}_6x6.png")
    print(f"\nSaved DSCF3561_{name}_6x6.png ({out.shape[1]}x{out.shape[0]})")

print("\nDone!")
