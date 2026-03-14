#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2024-present X-Veon contributors
"""HDR inference for CFA demosaicing with tile blending and post-demosaic white balance."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rawpy
import torch

from model import XTransUNet
from cfa import make_cfa_mask, make_channel_masks, detect_cfa_from_raw, find_pattern_shift, cfa_period, CFA_REGISTRY
from highlight_recovery import reconstruct_highlights
from highlight_recovery_rgb import reconstruct_highlights as reconstruct_highlights_rgb


# Standard color space conversion matrices
# XYZ to sRGB (D65 whitepoint)
XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
], dtype=np.float32)

# XYZ to BT.2020 (D65 whitepoint)
XYZ_TO_BT2020 = np.array([
    [ 1.7166512, -0.3556708, -0.2533663],
    [-0.6666844,  1.6164812,  0.0157685],
    [ 0.0176399, -0.0427706,  0.9421031]
], dtype=np.float32)


def apply_color_correction(rgb: np.ndarray, xyz_to_cam: np.ndarray = None,
                           to_bt2020: bool = True) -> np.ndarray:
    """Convert white-balanced camera RGB to sRGB or BT.2020.

    Uses dcraw's approach: build sRGB→Camera forward matrix, row-normalize
    in camera space, then invert to get Camera→sRGB.

    Args:
        rgb: (H, W, 3) linear camera RGB (white-balanced)
        xyz_to_cam: (3, 3) from rawpy.rgb_xyz_matrix — maps XYZ -> Camera (no WB)
        to_bt2020: if True, output BT.2020; if False, output sRGB

    Returns:
        (H, W, 3) linear sRGB or BT.2020 RGB
    """
    h, w, _ = rgb.shape
    rgb_flat = rgb.reshape(-1, 3)

    if xyz_to_cam is not None:
        # dcraw approach: normalize in forward direction, then invert
        # Step 1: sRGB→XYZ→Camera = sRGB→Camera (forward matrix)
        srgb_to_xyz = np.linalg.inv(XYZ_TO_SRGB.astype(np.float64))
        srgb_to_cam = xyz_to_cam.astype(np.float64) @ srgb_to_xyz

        # Step 2: Row-normalize per camera channel (dcraw convention)
        # Ensures sRGB white [1,1,1] → camera neutral [1,1,1]
        row_sums = srgb_to_cam.sum(axis=1, keepdims=True)
        srgb_to_cam = srgb_to_cam / row_sums

        # Step 3: Invert to get Camera→sRGB
        cam_to_srgb = np.linalg.inv(srgb_to_cam).astype(np.float32)

        # Step 4: For BT.2020, chain Camera→sRGB→BT.2020
        SRGB_TO_BT2020 = np.array([
            [0.6274039,  0.3292830,  0.0433131],
            [0.0690973,  0.9195404,  0.0113623],
            [0.0163914,  0.0880133,  0.8955953]
        ], dtype=np.float32)

        if to_bt2020:
            combined = SRGB_TO_BT2020 @ cam_to_srgb
        else:
            combined = cam_to_srgb

        result_full = rgb_flat @ combined.T
    else:
        # Fallback when no camera matrix available: assume camera RGB ~ sRGB
        if to_bt2020:
            SRGB_TO_BT2020 = np.array([
                [0.6274039,  0.3292830,  0.0433131],
                [0.0690973,  0.9195404,  0.0113623],
                [0.0163914,  0.0880133,  0.8955953]
            ], dtype=np.float32)
            result_full = rgb_flat @ SRGB_TO_BT2020.T
        else:
            result_full = rgb_flat

    return result_full.reshape(h, w, 3)


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


def linear_to_hlg(E: np.ndarray) -> np.ndarray:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    E = np.maximum(E, 0)
    return np.where(E <= 1/12, np.sqrt(3 * E), a * np.log(np.maximum(12 * E - b, 1e-10)) + c)


def extract_dr_gain(raw_path: str) -> float:
    """Extract Fuji DevelopmentDynamicRange from EXIF.
    Returns 1.0 (DR100), 2.0 (DR200), or 4.0 (DR400)."""
    try:
        import subprocess
        result = subprocess.run(
            ['exiftool', '-Fuji:DevelopmentDynamicRange', '-n', '-s3', raw_path],
            capture_output=True, text=True, timeout=5
        )
        value = int(result.stdout.strip())
        return value / 100.0
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        return 1.0


def process_raw(raw_path: str, model: torch.nn.Module, device: str,
                patch_size: int = 288, overlap: int = 48,
                apply_wb_to_cfa: bool = False,
                cfa_type: str | None = None,
                hlrecon: str = "cfa") -> tuple[np.ndarray, dict]:
    raw = rawpy.imread(raw_path)

    cfa = raw.raw_image_visible.astype(np.float32)
    black_per_ch = np.array(raw.black_level_per_channel, dtype=np.float32)
    white = float(raw.white_level)
    raw_pattern = raw.raw_colors_visible
    black = np.empty_like(cfa)
    for ch in range(4):
        black[raw_pattern == ch] = black_per_ch[ch]
    cfa_norm = (cfa - black) / (white - black)
    h_raw, w_raw = cfa_norm.shape

    # White balance multipliers (normalized to G=1)
    wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
    wb = wb / wb[1]

    # XYZ to Camera matrix (rawpy's rgb_xyz_matrix is actually XYZ→Camera)
    xyz_to_cam = np.array(raw.rgb_xyz_matrix[:3, :3], dtype=np.float32)

    # EXIF orientation
    exif_flip = raw.sizes.flip

    # Pattern alignment — auto-detect or use specified CFA type
    if cfa_type is not None:
        ref_pattern = CFA_REGISTRY[cfa_type]
    else:
        _, ref_pattern = detect_cfa_from_raw(raw_pattern)

    period = cfa_period(ref_pattern)
    dy, dx = find_pattern_shift(raw_pattern, ref_pattern)
    pad_top = (period - dy) % period
    pad_left = (period - dx) % period

    if hlrecon == "cfa":
        # darktable pipeline: WB → highlights → demosaic
        wb_map = np.ones_like(cfa_norm)
        for ch in range(3):
            wb_map[raw_pattern == ch] = wb[ch]
        cfa_norm = cfa_norm * wb_map

        clip_levels = wb
        cfa_norm = reconstruct_highlights(cfa_norm, raw_pattern, clip_levels)
    else:
        # No WB on CFA — clip at 1.0 (sensor max)
        clip_levels = np.ones(3, dtype=np.float32)

    if pad_top > 0 or pad_left > 0:
        cfa_norm = np.pad(cfa_norm, ((pad_top, 0), (pad_left, 0)), mode='reflect')

    h_aligned, w_aligned = cfa_norm.shape

    r_mask, g_mask, b_mask = make_channel_masks(patch_size, patch_size, ref_pattern)
    masks = torch.cat([r_mask.unsqueeze(0), g_mask.unsqueeze(0), b_mask.unsqueeze(0)], dim=0).to(device)

    # Per-pixel clip level map for one tile (CFA-aligned after padding, so periodic)
    tile_cfa = make_cfa_mask(patch_size, patch_size, ref_pattern).numpy()
    tile_clip_level = np.array([clip_levels[int(c)] for c in tile_cfa.flat],
                               dtype=np.float32).reshape(patch_size, patch_size)

    confidence_map = None
    variance = None

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
                    raw_ratio = np.clip(crop / (tile_clip_level + 1e-8), 0, 1)
                    clip_ratio = torch.from_numpy(np.clip((raw_ratio - 0.5) * 2.0, 0, 1).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                    inp = torch.cat([cfa_t, masks.unsqueeze(0), clip_ratio], dim=1)
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
        output_sq = np.zeros((3, h_pad, w_pad), dtype=np.float32)
        weights = np.zeros((h_pad, w_pad), dtype=np.float32)
        
        with torch.no_grad():
            for y in range(0, h_pad - patch_size + 1, stride):
                for x in range(0, w_pad - patch_size + 1, stride):
                    crop = cfa_padded[y:y+patch_size, x:x+patch_size]
                    cfa_t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float().to(device)
                    raw_ratio = np.clip(crop / (tile_clip_level + 1e-8), 0, 1)
                    clip_ratio = torch.from_numpy(np.clip((raw_ratio - 0.5) * 2.0, 0, 1).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                    inp = torch.cat([cfa_t, masks.unsqueeze(0), clip_ratio], dim=1)
                    out = model(inp)[0].cpu().numpy()

                    for c in range(3):
                        output[c, y:y+patch_size, x:x+patch_size] += out[c] * blend_weight
                        output_sq[c, y:y+patch_size, x:x+patch_size] += out[c]**2 * blend_weight
                    weights[y:y+patch_size, x:x+patch_size] += blend_weight
        
        weights = np.maximum(weights, 1e-8)
        output = output / weights[np.newaxis, :, :]
        mean_sq = output_sq / weights[np.newaxis, :, :]
        variance = np.maximum(mean_sq - output**2, 0)
    
    # Crop to original size
    rgb = output[:, pad_top:pad_top+h_raw, pad_left:pad_left+w_raw]

    if variance is not None:
        var_crop = variance[:, pad_top:pad_top+h_raw, pad_left:pad_left+w_raw]
        confidence_map = np.sqrt(var_crop.mean(axis=0))  # per-pixel RMSD across tiles

    rgb = rgb.transpose(1, 2, 0)

    if hlrecon == "rgb":
        # Post-demosaic path: apply WB to RGB, then highlight recovery
        rgb = rgb * wb[np.newaxis, np.newaxis, :]
        clip_levels_rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32) * wb
        rgb = reconstruct_highlights_rgb(rgb, clip_levels_rgb)

    # Color correction: camera RGB -> BT.2020
    rgb = apply_color_correction(rgb, xyz_to_cam=xyz_to_cam, to_bt2020=True)
    rgb = np.maximum(rgb, 0)

    raw.close()

    dr_gain = extract_dr_gain(raw_path)

    return rgb, {"exif_flip": exif_flip,
                  "confidence_map": confidence_map, "dr_gain": dr_gain}


# Backward compat
process_raf = process_raw


def save_hdr_avif(rgb: np.ndarray, output_path: str, quality: int = 90,
                  exif_flip: int = 0, dr_gain: float = 1.0):
    # Fuji DR compensation — undo deliberate underexposure
    if dr_gain > 1.0:
        rgb = rgb * dr_gain
        print(f"  DR gain: {dr_gain}x applied")

    # Apply EXIF rotation
    if exif_flip != 0:
        rgb = apply_exif_rotation(rgb, exif_flip)

    hlg = linear_to_hlg(rgb)
    hlg_u16 = (np.clip(hlg, 0, hlg.max()) / max(hlg.max(), 1.0) * 65535).astype(np.uint16)
    bgr_u16 = hlg_u16[:, :, ::-1]

    temp_png = output_path + ".tmp.png"
    cv2.imwrite(temp_png, bgr_u16)
    
    # CICP: 9 = BT.2020 primaries, 18 = HLG transfer, 9 = BT.2020 NCL
    avifenc = shutil.which("avifenc")
    if avifenc is None:
        raise FileNotFoundError("avifenc not found on PATH. Install libavif-bin (apt) or libavif (brew).")
    cmd = [avifenc, "-c", "aom", "-j", "all", "-s", "6",
            "-q", str(quality), "--cicp", "9/18/9", "-d", "10",
            "-y", "420", temp_png, output_path]
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
    parser.add_argument("--wb-cfa", action="store_true",
                        help="Apply WB to CFA before demosaic (for legacy checkpoints trained with --apply-wb)")
    parser.add_argument("--hlrecon", choices=["cfa", "rgb"], default="cfa",
                        help="Highlight reconstruction mode: cfa (pre-demosaic) or rgb (post-demosaic)")
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = XTransUNet(base_width=ckpt.get("base_width", 64))
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    ckpt_cfa = ckpt.get("cfa_type")
    print(f"Checkpoint: {args.checkpoint}" + (f" (cfa_type={ckpt_cfa})" if ckpt_cfa else ""))

    raw_globs = ["*.RAF", "*.raf", "*.CR2", "*.cr2", "*.CR3", "*.cr3",
                 "*.NEF", "*.nef", "*.ARW", "*.arw", "*.DNG", "*.dng"]

    if args.batch:
        input_dir = Path(args.input)
        output_dir = Path(args.output) if args.output else input_dir / "output"
        output_dir.mkdir(exist_ok=True)
        raw_files = []
        for ext in raw_globs:
            raw_files.extend(input_dir.glob(ext))
        for raw_file in raw_files:
            out_path = output_dir / f"{raw_file.stem}_hdr.avif"
            print(f"Processing {raw_file.name}...")
            rgb, meta = process_raw(str(raw_file), model, device, args.patch_size, args.overlap,
                                    apply_wb_to_cfa=args.wb_cfa, hlrecon=args.hlrecon)
            save_hdr_avif(rgb, str(out_path), args.quality,
                         exif_flip=meta.get("exif_flip", 0),
                         dr_gain=meta.get("dr_gain", 1.0))
    else:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.with_suffix(".avif")
        print(f"Processing {input_path.name}...")
        rgb, meta = process_raw(str(input_path), model, device, args.patch_size, args.overlap,
                                apply_wb_to_cfa=args.wb_cfa, hlrecon=args.hlrecon)
        save_hdr_avif(rgb, str(output_path), args.quality,
                     exif_flip=meta.get("exif_flip", 0),
                     dr_gain=meta.get("dr_gain", 1.0))
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
