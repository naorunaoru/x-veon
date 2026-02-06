#!/usr/bin/env python3
"""
Gradio web UI for X-Trans demosaicing inference.

Usage:
    python ui.py [--port 7860] [--share]
"""

import argparse
from glob import glob
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from model import XTransUNet
from infer_hdr import process_raf


# Global model cache
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
        return _model, _device
    
    _device = get_device()
    _model = XTransUNet().to(_device)
    
    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=True)
    _model.load_state_dict(ckpt["model"])
    _model.eval()
    _model_path = checkpoint_path
    
    epoch = ckpt.get("epoch", "?")
    psnr = ckpt.get("best_val_psnr", 0)
    print(f"Loaded {checkpoint_path}: epoch {epoch}, PSNR {psnr:.1f} dB")
    
    return _model, _device


def run_inference(
    raf_file,
    checkpoint: str,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[np.ndarray, str]:
    """Process RAF file and return demosaiced image."""
    
    if raf_file is None:
        raise gr.Error("Please upload a RAF file")
    
    if not checkpoint:
        raise gr.Error("Please select a checkpoint")
    
    progress(0.1, desc="Loading model...")
    model, device = load_model(checkpoint)
    
    progress(0.2, desc="Processing RAF (this takes 30-60s)...")
    raf_path = raf_file.name if hasattr(raf_file, 'name') else raf_file
    
    # Use the battle-tested process_raf from infer_hdr
    rgb_linear, meta = process_raf(raf_path, model, str(device), patch_size=288, overlap=48)
    
    progress(0.9, desc="Applying gamma...")
    
    # Clip and apply sRGB gamma for display
    rgb = np.clip(rgb_linear, 0, 1)
    rgb = np.power(rgb, 1/2.2)
    
    # Convert to uint8
    out_u8 = (rgb * 255).astype(np.uint8)
    
    progress(1.0, desc="Done!")
    
    # Status
    h, w = rgb_linear.shape[:2]
    ckpt_name = Path(checkpoint).parent.name + "/" + Path(checkpoint).name
    hdr_pixels = np.sum(rgb_linear > 1.0)
    status = f"{w}×{h} | {ckpt_name} | {hdr_pixels:,} HDR pixels clipped"
    
    return out_u8, status


def create_ui():
    """Create the Gradio interface."""
    
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
        
        def refresh_checkpoints():
            return gr.Dropdown(choices=find_checkpoints())
        
        refresh_btn.click(refresh_checkpoints, outputs=checkpoint_dropdown)
        
        process_btn.click(
            run_inference,
            inputs=[raf_input, checkpoint_dropdown],
            outputs=[output_image, status_text],
        )
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
