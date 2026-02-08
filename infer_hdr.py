#!/usr/bin/env python3
"""HDR inference for X-Trans demosaicing with tile blending and white balance."""

import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rawpy
import torch

from model import XTransUNet
from xtrans_pattern import make_channel_masks, XTRANS_PATTERN


# Fuji X-Trans camera RGB to sRGB matrix (estimated from rawpy's internal processing)
# Estimated from 10 images, 125k pixels, MAE=0.000035
# This matrix is for right-multiplication: srgb = cam_rgb @ FUJI_CAM_TO_SRGB
FUJI_CAM_TO_SRGB = np.array([
    [ 1.64032447, -0.18968946,  0.05009819],
    [-0.56355631,  1.64565551, -0.54180104],
    [-0.07882024, -0.45597395,  1.49148285]
], dtype=np.float32)

# sRGB to BT.2020 matrix (for HDR output)
SRGB_TO_BT2020 = np.array([
    [0.6274039,  0.3292830,  0.0433131],
    [0.0690973,  0.9195404,  0.0113623],
    [0.0163914,  0.0880133,  0.8955953]
], dtype=np.float32)


def apply_color_correction(rgb: np.ndarray, to_bt2020: bool = True) -> np.ndarray:
    """Convert camera RGB to sRGB or BT.2020.
    
    For highlights (luminance > threshold), blends towards identity matrix
    to prevent color shifts in blown areas.
    
    Args:
        rgb: (H, W, 3) linear camera RGB
        to_bt2020: if True, output BT.2020; if False, output sRGB
    
    Returns:
        (H, W, 3) linear sRGB or BT.2020 RGB
    """
    h, w, _ = rgb.shape
    rgb_flat = rgb.reshape(-1, 3)
    
    # Apply color matrix
    srgb_full = rgb_flat @ FUJI_CAM_TO_SRGB
    
    # For highlights, blend towards identity (no color change)
    # This keeps blown highlights neutral
    lum = 0.2126 * rgb_flat[:, 0] + 0.7152 * rgb_flat[:, 1] + 0.0722 * rgb_flat[:, 2]
    highlight_start = 0.7
    highlight_end = 1.0
    blend = np.clip((lum - highlight_start) / (highlight_end - highlight_start), 0, 1)
    blend = blend[:, np.newaxis]
    
    # Blend: full matrix for shadows/midtones, identity for highlights
    srgb = srgb_full * (1 - blend) + rgb_flat * blend
    
    if to_bt2020:
        # sRGB -> BT.2020
        result = srgb @ SRGB_TO_BT2020.T
    else:
        result = srgb
    
    return result.reshape(h, w, 3)


def apply_exif_rotation(img: np.ndarray, flip: int) -> np.ndarray:
    """Apply EXIF orientation to image.
    
    flip values from rawpy:
        0: no rotation
        3: 180°
        5: 90° CCW
        6: 90° CW
    """
    if flip == 0:
        return img
    elif flip == 3:
        return np.rot90(img, 2)
    elif flip == 5:
        return np.rot90(img, 1)  # 90° CCW
    elif flip == 6:
        return np.rot90(img, -1)  # 90° CW
    else:
        return img


def find_pattern_shift(raw_pattern: np.ndarray) -> tuple[int, int]:
    ref = np.array(XTRANS_PATTERN)
    for dy in range(6):
        for dx in range(6):
            shifted = np.roll(np.roll(ref, dy, axis=0), dx, axis=1)
            if np.array_equal(raw_pattern[:6, :6], shifted):
                return dy, dx
    raise ValueError("Could not match CFA pattern")


def linear_to_hlg(E: np.ndarray) -> np.ndarray:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    E = np.maximum(E, 0)
    return np.where(E <= 1/12, np.sqrt(3 * E), a * np.log(np.maximum(12 * E - b, 1e-10)) + c)


def process_raf(raf_path: str, model: torch.nn.Module, device: str,
                patch_size: int = 288, overlap: int = 48) -> tuple[np.ndarray, dict]:
    raw = rawpy.imread(raf_path)
    
    cfa = raw.raw_image_visible.astype(np.float32)
    black = raw.black_level_per_channel[0]
    white = raw.white_level
    cfa_norm = (cfa - black) / (white - black)
    h_raw, w_raw = cfa_norm.shape
    
    # White balance multipliers (normalized to G=1)
    wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
    wb = wb / wb[1]
    
    # Camera RGB to XYZ matrix
    cam_to_xyz = np.array(raw.rgb_xyz_matrix[:3, :3], dtype=np.float32)
    
    # EXIF orientation
    exif_flip = raw.sizes.flip
    
    # Pattern alignment
    raw_pattern = raw.raw_colors_visible
    dy, dx = find_pattern_shift(raw_pattern)
    pad_top = (6 - dy) % 6
    pad_left = (6 - dx) % 6
    
    if pad_top > 0 or pad_left > 0:
        cfa_norm = np.pad(cfa_norm, ((pad_top, 0), (pad_left, 0)), mode='reflect')
    
    h_aligned, w_aligned = cfa_norm.shape
    
    r_mask, g_mask, b_mask = make_channel_masks(patch_size, patch_size)
    masks = torch.cat([r_mask.unsqueeze(0), g_mask.unsqueeze(0), b_mask.unsqueeze(0)], dim=0).to(device)
    
    if overlap == 0:
        h_pad = ((h_aligned + patch_size - 1) // patch_size) * patch_size
        w_pad = ((w_aligned + patch_size - 1) // patch_size) * patch_size
        cfa_padded = np.zeros((h_pad, w_pad), dtype=np.float32)
        cfa_padded[:h_aligned, :w_aligned] = cfa_norm
        
        output = np.zeros((3, h_pad, w_pad), dtype=np.float32)
        
        with torch.no_grad():
            for y in range(0, h_pad, patch_size):
                for x in range(0, w_pad, patch_size):
                    crop = cfa_padded[y:y+patch_size, x:x+patch_size]
                    cfa_t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float().to(device)
                    inp = torch.cat([cfa_t, masks.unsqueeze(0)], dim=1)
                    out = model(inp)[0].cpu().numpy()
                    output[:, y:y+patch_size, x:x+patch_size] = out
    else:
        stride = patch_size - overlap
        h_pad = ((h_aligned - overlap + stride - 1) // stride) * stride + patch_size
        w_pad = ((w_aligned - overlap + stride - 1) // stride) * stride + patch_size
        
        cfa_padded = np.zeros((h_pad, w_pad), dtype=np.float32)
        cfa_padded[:h_aligned, :w_aligned] = cfa_norm
        
        weight_1d = np.ones(patch_size, dtype=np.float32)
        weight_1d[:overlap] = np.linspace(0, 1, overlap)
        weight_1d[-overlap:] = np.linspace(1, 0, overlap)
        blend_weight = np.outer(weight_1d, weight_1d)
        
        output = np.zeros((3, h_pad, w_pad), dtype=np.float32)
        weights = np.zeros((h_pad, w_pad), dtype=np.float32)
        
        with torch.no_grad():
            for y in range(0, h_pad - patch_size + 1, stride):
                for x in range(0, w_pad - patch_size + 1, stride):
                    crop = cfa_padded[y:y+patch_size, x:x+patch_size]
                    cfa_t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float().to(device)
                    inp = torch.cat([cfa_t, masks.unsqueeze(0)], dim=1)
                    out = model(inp)[0].cpu().numpy()
                    
                    for c in range(3):
                        output[c, y:y+patch_size, x:x+patch_size] += out[c] * blend_weight
                    weights[y:y+patch_size, x:x+patch_size] += blend_weight
        
        weights = np.maximum(weights, 1e-8)
        output = output / weights[np.newaxis, :, :]
    
    # Crop to original size
    rgb = output[:, pad_top:pad_top+h_raw, pad_left:pad_left+w_raw]
    
    # Don't apply WB here - do it after color matrix conversion
    rgb = rgb.transpose(1, 2, 0)
    raw.close()
    
    return rgb, {"wb": wb, "cam_to_xyz": cam_to_xyz, "exif_flip": exif_flip}


def save_hdr_avif(rgb: np.ndarray, output_path: str, quality: int = 90, 
                  cam_to_xyz: np.ndarray = None, exif_flip: int = 0,
                  wb: np.ndarray = None, apply_color: bool = True):
    # Apply white balance
    if wb is not None:
        rgb = rgb * wb[np.newaxis, np.newaxis, :]
    
    # Apply color correction: camera RGB -> sRGB -> BT.2020
    if apply_color:
        rgb = apply_color_correction(rgb, to_bt2020=True)
        rgb = np.maximum(rgb, 0)  # Clip negative values from matrix math
    
    # Apply EXIF rotation
    if exif_flip != 0:
        rgb = apply_exif_rotation(rgb, exif_flip)
    
    hlg = linear_to_hlg(rgb)
    hlg_u16 = (np.clip(hlg, 0, 1) * 65535).astype(np.uint16)
    bgr_u16 = hlg_u16[:, :, ::-1]
    
    temp_png = output_path + ".tmp.png"
    cv2.imwrite(temp_png, bgr_u16)
    
    # Use full path for homebrew avifenc (needed when running without shell PATH)
    # CICP: 9 = BT.2020 primaries, 18 = HLG transfer, 9 = BT.2020 NCL
    avifenc = "/opt/homebrew/bin/avifenc"
    cmd = [avifenc, "--min", "0", "--max", "63", "-q", str(quality),
           "--cicp", "9/18/9", "-d", "10", temp_png, output_path]
    subprocess.run(cmd, check=True, capture_output=True)
    os.remove(temp_png)
    
    over_1 = np.sum(rgb > 1.0)
    print(f"  HDR: {over_1:,} pixels > 1.0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints_v4_ft/best.pt")
    parser.add_argument("--patch-size", type=int, default=288)
    parser.add_argument("--overlap", type=int, default=48)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--no-color", action="store_true", help="Skip color correction")
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = XTransUNet()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"Checkpoint: {args.checkpoint}")
    
    if args.batch:
        input_dir = Path(args.input)
        output_dir = Path(args.output) if args.output else input_dir / "output"
        output_dir.mkdir(exist_ok=True)
        for raf in list(input_dir.glob("*.RAF")) + list(input_dir.glob("*.raf")):
            out_path = output_dir / f"{raf.stem}_hdr.avif"
            print(f"Processing {raf.name}...")
            rgb, meta = process_raf(str(raf), model, device, args.patch_size, args.overlap)
            exif_flip = meta.get("exif_flip", 0)
            wb = meta.get("wb")
            save_hdr_avif(rgb, str(out_path), args.quality, None, exif_flip, wb, not args.no_color)
    else:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.with_suffix(".avif")
        print(f"Processing {input_path.name}...")
        rgb, meta = process_raf(str(input_path), model, device, args.patch_size, args.overlap)
        exif_flip = meta.get("exif_flip", 0)
        wb = meta.get("wb")
        save_hdr_avif(rgb, str(output_path), args.quality, None, exif_flip, wb, not args.no_color)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
