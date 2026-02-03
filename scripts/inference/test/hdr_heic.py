#!/usr/bin/env python3
"""
HDR HEIC output with proper P3 + PQ metadata.
"""

import numpy as np
from PIL import Image
import pillow_heif
from pillow_heif import HeifColorPrimaries, HeifTransferCharacteristics

pillow_heif.register_heif_opener()

# XYZ to Display P3 matrix (D65)
XYZ_TO_P3 = np.array([
    [ 2.4934969, -0.9313836, -0.4027108],
    [-0.8294890,  1.7626641,  0.0236247],
    [ 0.0358458, -0.0761724,  0.9568845]
])

# PQ (SMPTE ST 2084) constants
PQ_M1 = 0.1593017578125
PQ_M2 = 78.84375
PQ_C1 = 0.8359375
PQ_C2 = 18.8515625
PQ_C3 = 18.6875

def linear_to_pq(linear, peak_nits=1000):
    """Convert linear light (0-1 = 0-peak_nits) to PQ signal."""
    L = np.clip(linear * peak_nits / 10000, 1e-10, 1)
    Lm1 = np.power(L, PQ_M1)
    pq = np.power((PQ_C1 + PQ_C2 * Lm1) / (1 + PQ_C3 * Lm1), PQ_M2)
    return pq

def camera_rgb_to_p3(rgb, cam_to_xyz):
    """Convert camera RGB to Display P3."""
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    xyz = flat @ cam_to_xyz[:3, :3].T
    p3 = xyz @ XYZ_TO_P3.T
    return p3.reshape(h, w, 3)

def tonemap_reinhard(rgb, white_point=2.0):
    """Simple Reinhard tonemapping: L / (1 + L)"""
    # Convert to luminance
    lum = 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]
    lum = np.maximum(lum, 1e-10)
    
    # Reinhard operator
    lum_mapped = lum * (1 + lum / (white_point ** 2)) / (1 + lum)
    
    # Scale RGB by luminance ratio
    scale = lum_mapped / lum
    return rgb * scale[:,:,np.newaxis]

def save_hdr_heic(linear_rgb, output_path, cam_to_xyz, peak_nits=1000, quality=90):
    """
    Save linear RGB as HDR HEIC with P3 + PQ metadata.
    
    linear_rgb: (H, W, 3) float32 in linear light, white-balanced
    cam_to_xyz: camera RGB to XYZ matrix
    peak_nits: display peak brightness for PQ encoding
    """
    print(f"Input range: [{linear_rgb.min():.4f}, {linear_rgb.max():.4f}]")
    
    # Clip negative values (from model artifacts)
    linear_rgb = np.maximum(linear_rgb, 0)
    
    # Camera RGB -> Display P3 (linear)
    p3_linear = camera_rgb_to_p3(linear_rgb, cam_to_xyz)
    print(f"P3 linear range: [{p3_linear.min():.4f}, {p3_linear.max():.4f}]")
    
    # Gamut clip negative P3 values
    p3_linear = np.maximum(p3_linear, 0)
    
    # Tonemap highlights (preserve HDR but prevent extreme values)
    if p3_linear.max() > 1.0:
        p3_linear = tonemap_reinhard(p3_linear, white_point=p3_linear.max())
        print(f"After tonemap: [{p3_linear.min():.4f}, {p3_linear.max():.4f}]")
    
    # Apply PQ transfer function
    pq = linear_to_pq(p3_linear, peak_nits)
    print(f"PQ range: [{pq.min():.4f}, {pq.max():.4f}]")
    
    # Convert to 8-bit (pillow-heif limitation for now)
    pq_8bit = np.clip(pq * 255, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(pq_8bit, mode='RGB')
    
    # Save with HDR metadata
    img.save(
        output_path,
        format='HEIF',
        quality=quality,
        save_nclx_profile=True,
        color_primaries=HeifColorPrimaries.SMPTE_EG_432_1.value,  # Display P3
        transfer_characteristics=HeifTransferCharacteristics.ITU_R_BT_2100_0_PQ.value,  # PQ
        matrix_coefficients=0,  # Identity (RGB, not YCbCr)
        full_range_flag=1
    )
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    print("Creating test HDR gradient...")
    
    h, w = 512, 512
    x = np.linspace(0, 4, w)  # 0 to 400% brightness (HDR)
    y = np.linspace(0, 1, h)
    
    r = np.outer(y, x)
    g = np.outer(y, x) * 0.8
    b = np.outer(y, x) * 0.5
    
    linear_rgb = np.stack([r, g, b], axis=2).astype(np.float32)
    identity = np.eye(3)
    
    save_hdr_heic(linear_rgb, "test_hdr_gradient.heic", identity, peak_nits=1000)
    print("Done!")
