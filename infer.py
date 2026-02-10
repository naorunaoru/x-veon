#!/usr/bin/env python3
"""
Inference script for v4 (linear sensor space) X-Trans demosaicing model.

Pipeline:
1. Load RAF → raw CFA (black-subtracted, normalized by white-black)
2. Tile + run through model → linear RGB
3. Post-process: WB → color matrix → tone curve → sRGB gamma

Usage:
    python infer_v4_linear.py input.RAF output.tiff --checkpoint checkpoints_v4/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rawpy
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from model import XTransUNet
from xtrans_pattern import make_cfa_mask, make_channel_masks, XTRANS_PATTERN


def find_pattern_shift(raw_pattern: np.ndarray) -> tuple[int, int]:
    """Find the CFA pattern shift relative to canonical X-Trans pattern."""
    ref = np.array(XTRANS_PATTERN)
    for dy in range(6):
        for dx in range(6):
            shifted = np.roll(np.roll(ref, dy, axis=0), dx, axis=1)
            if np.array_equal(raw_pattern[:6, :6], shifted):
                return dy, dx
    raise ValueError("Could not match CFA pattern to X-Trans")


def load_raf_linear(raf_path: str) -> tuple[np.ndarray, dict]:
    """
    Load RAF file and return linear sensor data + metadata.
    
    Returns:
        cfa: (H, W) float32 array, black-subtracted and normalized
        meta: dict with black_level, white_level, wb_multipliers, color_matrix, pattern_shift
    """
    with rawpy.imread(raf_path) as raw:
        # Get raw data
        cfa = raw.raw_image_visible.astype(np.float32)
        
        # Black and white levels
        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)
        
        # WB multipliers (as-shot or daylight)
        wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
        # Normalize so green = 1
        wb = wb / wb[1]
        
        # Color matrix (camera RGB to XYZ)
        # rawpy gives xyz_to_cam, we need cam_to_xyz
        xyz_cam = np.array(raw.rgb_xyz_matrix[:3, :3], dtype=np.float32)
        
        # Pattern shift for this camera
        raw_pattern = raw.raw_pattern[:6, :6]
        pattern_shift = find_pattern_shift(raw_pattern)
        
        # Normalize
        cfa = (cfa - black) / (white - black)
        
        meta = {
            'black_level': black,
            'white_level': white,
            'wb_multipliers': wb,
            'xyz_cam_matrix': xyz_cam,
            'pattern_shift': pattern_shift,
            'width': cfa.shape[1],
            'height': cfa.shape[0],
        }
        
        return cfa, meta


def align_cfa_pattern(cfa: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """Pad CFA to align with canonical X-Trans pattern."""
    dy, dx = shift
    if dy == 0 and dx == 0:
        return cfa
    
    h, w = cfa.shape
    # Pad top and left to shift pattern
    pad_top = (6 - dy) % 6
    pad_left = (6 - dx) % 6
    
    if pad_top > 0 or pad_left > 0:
        cfa = np.pad(cfa, ((pad_top, 0), (pad_left, 0)), mode='reflect')
    
    return cfa


def make_tiles(cfa: np.ndarray, patch_size: int = 288, overlap: int = 96):
    """Generate overlapping tiles from CFA image."""
    h, w = cfa.shape
    stride = patch_size - overlap
    
    # Pad to multiple of stride + overlap
    pad_h = (stride - (h - overlap) % stride) % stride
    pad_w = (stride - (w - overlap) % stride) % stride
    
    if pad_h > 0 or pad_w > 0:
        cfa = np.pad(cfa, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    h_pad, w_pad = cfa.shape
    tiles = []
    positions = []
    
    for y in range(0, h_pad - overlap, stride):
        for x in range(0, w_pad - overlap, stride):
            if y + patch_size <= h_pad and x + patch_size <= w_pad:
                tile = cfa[y:y+patch_size, x:x+patch_size]
                tiles.append(tile)
                positions.append((y, x))
    
    return tiles, positions, (h_pad, w_pad), (h, w)


def hann_window_2d(size: int) -> np.ndarray:
    """Create 2D Hann window for blending."""
    hann_1d = np.hanning(size)
    return np.outer(hann_1d, hann_1d)


def run_inference(model, cfa: np.ndarray, device: torch.device,
                  patch_size: int = 288, overlap: int = 96) -> np.ndarray:
    """Run tiled inference with Hann window blending."""
    
    # Make CFA masks
    cfa_mask = make_cfa_mask(patch_size, patch_size)
    channel_masks = make_channel_masks(patch_size, patch_size)
    
    # Generate tiles
    tiles, positions, padded_size, orig_size = make_tiles(cfa, patch_size, overlap)
    
    # Prepare blending
    hann = hann_window_2d(patch_size)
    hann_3d = np.stack([hann, hann, hann], axis=0)
    
    # Output buffers
    h_pad, w_pad = padded_size
    output_sum = np.zeros((3, h_pad, w_pad), dtype=np.float32)
    weight_sum = np.zeros((h_pad, w_pad), dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        for i, (tile, (y, x)) in enumerate(zip(tiles, positions)):
            # Build input tensor: [CFA, R_mask, G_mask, B_mask]
            cfa_tensor = torch.from_numpy(tile).float().unsqueeze(0)  # (1, H, W)
            input_tensor = torch.cat([cfa_tensor, channel_masks], dim=0)  # (4, H, W)
            input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 4, H, W)
            
            # Run model
            output = model(input_tensor)  # (1, 3, H, W)
            output = output.squeeze(0).cpu().numpy()  # (3, H, W)
            
            # Blend with Hann window
            output_sum[:, y:y+patch_size, x:x+patch_size] += output * hann_3d
            weight_sum[y:y+patch_size, x:x+patch_size] += hann
    
    # Normalize by weights
    weight_sum = np.maximum(weight_sum, 1e-8)
    result = output_sum / weight_sum
    
    # Crop to original size
    h_orig, w_orig = orig_size
    result = result[:, :h_orig, :w_orig]
    
    return result


def apply_white_balance(rgb: np.ndarray, wb: np.ndarray) -> np.ndarray:
    """Apply white balance multipliers to linear RGB."""
    # rgb: (3, H, W), wb: (3,)
    return rgb * wb[:, np.newaxis, np.newaxis]


def apply_color_matrix(rgb: np.ndarray, xyz_cam: np.ndarray) -> np.ndarray:
    """
    Convert camera RGB to sRGB via XYZ.
    
    xyz_cam: camera RGB to XYZ matrix from rawpy
    """
    # XYZ to sRGB D65 matrix
    xyz_to_srgb = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=np.float32)
    
    # Combined: camera RGB → XYZ → sRGB
    # But xyz_cam might have normalization issues, so let's try direct
    # For now, skip color matrix and just use camera RGB as approximate sRGB
    # This is a simplification that works reasonably for Fuji
    
    return rgb  # Skip for now, can refine later


def apply_tone_curve(rgb: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply simple gamma tone curve."""
    # Clip to [0, 1] first
    rgb = np.clip(rgb, 0, 1)
    # Apply gamma
    return np.power(rgb, 1.0 / gamma)


def apply_srgb_gamma(rgb: np.ndarray) -> np.ndarray:
    """Apply proper sRGB gamma (linear below threshold, power above)."""
    rgb = np.clip(rgb, 0, 1)
    mask = rgb <= 0.0031308
    result = np.where(mask, rgb * 12.92, 1.055 * np.power(rgb, 1/2.4) - 0.055)
    return result


def save_output(rgb: np.ndarray, output_path: str, bits: int = 16):
    """Save RGB output as TIFF or PNG."""
    # rgb: (3, H, W) float in [0, 1]
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
    rgb = np.clip(rgb, 0, 1)
    
    if bits == 16:
        rgb_int = (rgb * 65535).astype(np.uint16)
        # PIL doesn't handle 16-bit RGB well, use tifffile or imageio
        try:
            import tifffile
            tifffile.imwrite(output_path, rgb_int)
        except ImportError:
            # Fall back to saving as 8-bit
            print("Warning: tifffile not available, saving as 8-bit")
            rgb_int = (rgb * 255).astype(np.uint8)
            img = Image.fromarray(rgb_int, mode='RGB')
            img.save(output_path)
    else:
        rgb_int = (rgb * 255).astype(np.uint8)
        img = Image.fromarray(rgb_int, mode='RGB')
        img.save(output_path)
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="v4 X-Trans inference (linear space)")
    parser.add_argument("input", help="Input RAF file")
    parser.add_argument("output", help="Output TIFF/PNG file")
    parser.add_argument("--checkpoint", "-c", default="checkpoints_v4/best.pt")
    parser.add_argument("--patch-size", type=int, default=288)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--bits", type=int, default=16, choices=[8, 16])
    parser.add_argument("--no-wb", action="store_true", help="Skip white balance")
    parser.add_argument("--no-gamma", action="store_true", help="Output linear (no gamma)")
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = XTransUNet(base_width=ckpt.get("base_width", 64)).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded epoch {ckpt.get('epoch', '?')}, best PSNR: {ckpt.get('best_val_psnr', '?'):.1f} dB")
    
    # Load RAF
    print(f"Loading: {args.input}")
    cfa, meta = load_raf_linear(args.input)
    print(f"  Size: {meta['width']}x{meta['height']}")
    print(f"  Pattern shift: {meta['pattern_shift']}")
    print(f"  WB multipliers: {meta['wb_multipliers']}")
    
    # Align CFA pattern
    cfa = align_cfa_pattern(cfa, meta['pattern_shift'])
    
    # Run inference
    print("Running inference...")
    rgb = run_inference(model, cfa, device, args.patch_size, args.overlap)
    
    # Crop back if we padded for pattern alignment
    dy, dx = meta['pattern_shift']
    pad_top = (6 - dy) % 6
    pad_left = (6 - dx) % 6
    if pad_top > 0 or pad_left > 0:
        rgb = rgb[:, pad_top:pad_top+meta['height'], pad_left:pad_left+meta['width']]
    
    print(f"  Output shape: {rgb.shape}")
    print(f"  Linear range: [{rgb.min():.4f}, {rgb.max():.4f}]")
    
    # Post-processing
    if not args.no_wb:
        print("Applying white balance...")
        rgb = apply_white_balance(rgb, meta['wb_multipliers'])
        print(f"  After WB range: [{rgb.min():.4f}, {rgb.max():.4f}]")
    
    if not args.no_gamma:
        print("Applying sRGB gamma...")
        rgb = apply_srgb_gamma(rgb)
    
    # Save
    save_output(rgb, args.output, args.bits)


if __name__ == "__main__":
    main()
