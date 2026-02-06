#!/usr/bin/env python3
"""
Gradio web UI for X-Trans demosaicing inference.

Usage:
    python ui.py [--port 7860] [--share]

Access at http://localhost:7860 (or http://macmini.local:7860 from LAN)
"""

import argparse
from glob import glob
from pathlib import Path

import gradio as gr
import numpy as np
import rawpy
import torch

from model import XTransUNet
from xtrans_pattern import make_channel_masks, XTRANS_PATTERN


# Global state
_model = None
_model_path = None
_device = None


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_checkpoints():
    """Find all available checkpoints."""
    patterns = [
        "checkpoints*/best.pt",
        "checkpoints*/latest.pt",
        "checkpoints_archive/*.pt",
    ]
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob(pattern))
    return sorted(set(checkpoints), reverse=True)


def load_model(checkpoint_path: str):
    """Load model from checkpoint (with caching)."""
    global _model, _model_path, _device
    
    if _model_path == checkpoint_path and _model is not None:
        return _model
    
    _device = get_device()
    _model = XTransUNet().to(_device)
    
    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=True)
    _model.load_state_dict(ckpt["model"])
    _model.eval()
    _model_path = checkpoint_path
    
    epoch = ckpt.get("epoch", "?")
    psnr = ckpt.get("best_val_psnr", 0)
    print(f"Loaded {checkpoint_path}: epoch {epoch}, PSNR {psnr:.1f} dB")
    
    return _model


def find_pattern_shift(raw_pattern: np.ndarray) -> tuple[int, int]:
    """Find CFA pattern shift relative to canonical X-Trans."""
    ref = np.array(XTRANS_PATTERN)
    for dy in range(6):
        for dx in range(6):
            shifted = np.roll(np.roll(ref, dy, axis=0), dx, axis=1)
            if np.array_equal(raw_pattern[:6, :6], shifted):
                return dy, dx
    raise ValueError("Could not match CFA pattern")


def process_raf(
    raf_file,
    checkpoint: str,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[np.ndarray, str]:
    """Process RAF file and return demosaiced image."""
    
    if raf_file is None:
        raise gr.Error("Please upload a RAF file")
    
    if not checkpoint:
        raise gr.Error("Please select a checkpoint")
    
    progress(0, desc="Loading model...")
    model = load_model(checkpoint)
    device = _device
    
    progress(0.1, desc="Loading RAF...")
    raf_path = raf_file.name if hasattr(raf_file, 'name') else raf_file
    
    with rawpy.imread(raf_path) as raw:
        # Get raw CFA data
        cfa = raw.raw_image_visible.astype(np.float32)
        black = np.array(raw.black_level_per_channel).mean()
        white = raw.white_level
        cfa = (cfa - black) / (white - black)
        
        # Get pattern shift
        pattern_shift = find_pattern_shift(raw.raw_pattern)
        
        # Get white balance
        wb = np.array(raw.camera_whitebalance[:3])
        wb = wb / wb[1]  # Normalize to green=1
    
    orig_h, orig_w = cfa.shape
    
    # Pad for pattern alignment
    dy, dx = pattern_shift
    if dy > 0 or dx > 0:
        cfa = np.pad(cfa, ((dy, 0), (dx, 0)), mode='reflect')
    
    # Pad for 6-pixel alignment
    h, w = cfa.shape
    pad_h = (6 - h % 6) % 6
    pad_w = (6 - w % 6) % 6
    if pad_h or pad_w:
        cfa = np.pad(cfa, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    h, w = cfa.shape
    
    # Tiled inference with Hann blending
    patch_size = 288
    overlap = 96
    step = patch_size - overlap
    
    out = np.zeros((h, w, 3), dtype=np.float32)
    weights = np.zeros((h, w, 1), dtype=np.float32)
    
    hann_1d = np.hanning(patch_size).astype(np.float32)
    hann_2d = np.outer(hann_1d, hann_1d)[:, :, None]
    
    # Prepare channel masks
    masks = make_channel_masks(patch_size, patch_size).numpy()
    
    # Count tiles for progress
    tiles_y = len(range(0, h - patch_size + 1, step))
    tiles_x = len(range(0, w - patch_size + 1, step))
    total_tiles = tiles_y * tiles_x
    
    progress(0.2, desc=f"Processing {total_tiles} tiles...")
    
    tile_idx = 0
    with torch.no_grad():
        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                patch = cfa[y:y+patch_size, x:x+patch_size]
                
                # Build input: [CFA, R_mask, G_mask, B_mask]
                inp = np.concatenate([patch[None], masks], axis=0)
                inp = torch.from_numpy(inp).unsqueeze(0).to(device)
                
                pred = model(inp)
                pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                out[y:y+patch_size, x:x+patch_size] += pred * hann_2d
                weights[y:y+patch_size, x:x+patch_size] += hann_2d
                
                tile_idx += 1
                if tile_idx % 10 == 0:
                    progress(0.2 + 0.7 * tile_idx / total_tiles, 
                            desc=f"Tile {tile_idx}/{total_tiles}")
    
    progress(0.9, desc="Finalizing...")
    
    # Blend
    out = out / np.maximum(weights, 1e-8)
    
    # Remove padding
    if dy > 0 or dx > 0:
        out = out[dy:, dx:]
    out = out[:orig_h, :orig_w]
    
    # Post-processing: white balance + gamma
    out = out * wb[None, None, :]
    out = np.clip(out, 0, 1)
    out = np.power(out, 1/2.2)  # sRGB gamma
    
    # Convert to uint8 for display
    out_u8 = (out * 255).astype(np.uint8)
    
    progress(1.0, desc="Done!")
    
    # Status message
    ckpt_name = Path(checkpoint).parent.name + "/" + Path(checkpoint).name
    status = f"Processed {orig_w}×{orig_h} with {ckpt_name} ({total_tiles} tiles)"
    
    return out_u8, status


def create_ui():
    """Create and return the Gradio interface."""
    
    checkpoints = find_checkpoints()
    default_ckpt = checkpoints[0] if checkpoints else None
    
    with gr.Blocks(title="X-Trans Demosaic") as demo:
        gr.Markdown("# X-Trans Demosaicing")
        gr.Markdown("Upload a Fujifilm RAF file to demosaic with the neural network model.")
        
        with gr.Row():
            with gr.Column(scale=1):
                raf_input = gr.File(
                    label="RAF File",
                    file_types=[".RAF", ".raf"],
                    type="filepath",
                )
                
                checkpoint_dropdown = gr.Dropdown(
                    choices=checkpoints,
                    value=default_ckpt,
                    label="Checkpoint",
                    allow_custom_value=True,
                )
                
                refresh_btn = gr.Button("🔄 Refresh Checkpoints", size="sm")
                process_btn = gr.Button("Process", variant="primary")
                
                status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=2):
                output_image = gr.Image(label="Output", type="numpy")
        
        # Refresh checkpoints list
        def refresh_checkpoints():
            new_checkpoints = find_checkpoints()
            return gr.Dropdown(choices=new_checkpoints)
        
        refresh_btn.click(refresh_checkpoints, outputs=checkpoint_dropdown)
        
        # Process
        process_btn.click(
            process_raf,
            inputs=[raf_input, checkpoint_dropdown],
            outputs=[output_image, status_text],
        )
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",  # Allow LAN access
        server_port=args.port,
        share=args.share,
    )
