#!/usr/bin/env python3
"""
Gradio web UI for X-Trans demosaicing inference.

Usage:
    python ui.py [--port 7860] [--share]
"""

import argparse
import base64
import json
import tempfile
from glob import glob
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import XTransUNet
from infer_hdr import apply_exif_rotation, process_raf, save_hdr_avif


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


def find_checkpoint_dirs():
    """Find all checkpoint directories with history.json."""
    dirs = sorted(glob("checkpoints*"))
    return [d for d in dirs if Path(d, "history.json").exists()]


def load_history(checkpoint_dir: str) -> list[dict]:
    """Load training history from checkpoint directory."""
    history_path = Path(checkpoint_dir) / "history.json"
    if not history_path.exists():
        return []
    with open(history_path) as f:
        return json.load(f)


def plot_training_history(checkpoint_dir: str) -> tuple:
    """Generate training history plots."""
    history = load_history(checkpoint_dir)
    if not history:
        return None, "No history found"
    
    epochs = [h["epoch"] for h in history]
    train_psnr = [h["train_psnr"] for h in history]
    val_psnr = [h["val_psnr"] for h in history]
    
    # Load config for title
    config_path = Path(checkpoint_dir) / "config.json"
    config_str = ""
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        parts = []
        if cfg.get("msssim_weight"): parts.append(f"MS-SSIM={cfg['msssim_weight']}")
        if cfg.get("per_channel_norm"): parts.append("per-ch-norm")
        if cfg.get("color_bias_weight"): parts.append(f"color_bias={cfg['color_bias_weight']}")
        if cfg.get("torture_fraction"): parts.append(f"torture={cfg['torture_fraction']*100:.0f}%")
        if cfg.get("apply_wb"): parts.append("WB")
        config_str = ", ".join(parts)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1, ax2, ax3 = axes
    
    # Smoothed validation (EMA)
    def ema(data, alpha=0.1):
        smoothed = [data[0]]
        for v in data[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
        return smoothed
    
    # PSNR plot
    ax1.plot(epochs, train_psnr, label="Train", alpha=0.5)
    ax1.plot(epochs, val_psnr, label="Val", alpha=0.3, linewidth=1)
    val_smoothed = ema(val_psnr, alpha=0.1)
    ax1.plot(epochs, val_smoothed, label="Val (smoothed)", linewidth=2, color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"{checkpoint_dir}\n{config_str}" if config_str else checkpoint_dir)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Best PSNR annotation
    best_idx = np.argmax(val_psnr)
    ax1.annotate(f"Best: {val_psnr[best_idx]:.2f} dB\n(epoch {epochs[best_idx]})",
                xy=(epochs[best_idx], val_psnr[best_idx]),
                xytext=(10, -20), textcoords="offset points",
                fontsize=9, color="green",
                arrowprops=dict(arrowstyle="->", color="green", alpha=0.7))
    
    # Helper to plot components
    def plot_components(ax, history, key, title):
        if key not in history[0]:
            return
        # Support both l1 and huber
        components = ["l1", "huber", "msssim", "gradient", "chroma", "color_bias"]
        for comp in components:
            values = [h[key].get(comp, 0) for h in history]
            if any(v > 0 for v in values):
                # Convert MS-SSIM to loss (it's stored as similarity)
                if comp == "msssim":
                    values = [1 - v for v in values]
                ax.plot(epochs, values, label=comp, alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    
    # Train components
    plot_components(ax2, history, "train_components", "Train Components")
    
    # Val components
    plot_components(ax3, history, "val_components", "Val Components")
    
    plt.tight_layout()
    
    # Current status
    latest = history[-1]
    status = f"Epoch {latest['epoch']}/{cfg.get('epochs', '?')} | Val PSNR: {latest['val_psnr']:.2f} dB | Best: {val_psnr[best_idx]:.2f} dB (ep {epochs[best_idx]})"
    
    return fig, status


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


def make_confidence_heatmap(confidence_map: np.ndarray, exif_flip: int = 0) -> tuple[np.ndarray, str]:
    """Convert confidence map to a colored heatmap image and stats string."""
    if exif_flip != 0:
        confidence_map = apply_exif_rotation(confidence_map, exif_flip)

    p99 = np.percentile(confidence_map, 99)
    normalized = np.clip(confidence_map / max(p99, 1e-8), 0, 1)

    cmap = plt.cm.inferno
    heatmap = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)

    mean_val = confidence_map.mean()
    max_val = confidence_map.max()
    high_pct = (confidence_map > p99).mean() * 100
    stats = f"mean={mean_val:.4f} | max={max_val:.4f} | p99={p99:.4f} | >{p99:.4f}: {high_pct:.1f}%"

    return heatmap, stats


def run_inference(
    raf_file,
    checkpoint: str,
    overlap: int = 48,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str, str, str, np.ndarray | None, str]:
    """Process RAF file and return HDR AVIF."""

    if raf_file is None:
        raise gr.Error("Please upload a RAF file")

    if not checkpoint:
        raise gr.Error("Please select a checkpoint")

    progress(0.1, desc="Loading model...")
    model, device = load_model(checkpoint)

    patch_size = 288
    stride = patch_size - overlap
    progress(0.2, desc=f"Demosaicing (overlap={overlap}, stride={stride})...")
    raf_path = raf_file.name if hasattr(raf_file, 'name') else raf_file
    raf_name = Path(raf_path).stem

    rgb_linear, meta = process_raf(raf_path, model, str(device), patch_size=patch_size, overlap=overlap)

    progress(0.9, desc="Encoding HDR AVIF...")

    output_path = tempfile.mktemp(suffix=".avif", prefix=f"{raf_name}_hdr_")
    save_hdr_avif(rgb_linear, output_path, 90, meta.get("xyz_to_cam"), meta.get("exif_flip", 0),
                  None, True)

    # Read and base64 encode for HTML display
    with open(output_path, "rb") as f:
        avif_b64 = base64.b64encode(f.read()).decode()

    # Confidence heatmap
    conf_map = meta.get("confidence_map")
    heatmap_img, conf_stats = None, ""
    if conf_map is not None:
        heatmap_img, conf_stats = make_confidence_heatmap(conf_map, meta.get("exif_flip", 0))

    progress(1.0, desc="Done!")

    # Status
    h, w = rgb_linear.shape[:2]
    ckpt_name = Path(checkpoint).parent.name + "/" + Path(checkpoint).name
    hdr_pixels = np.sum(rgb_linear > 1.0)
    status = f"{w}×{h} | {ckpt_name} | {hdr_pixels:,} HDR pixels"
    
    # HTML with fullscreen support
    html = f'''
    <style>
        .hdr-container {{ position: relative; width: 100%; }}
        .hdr-container img {{ 
            width: 100%; 
            cursor: pointer;
            border-radius: 8px;
        }}
        .hdr-container img:fullscreen {{ 
            object-fit: contain;
            background: black;
        }}
        .fullscreen-hint {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
        }}
    </style>
    <div class="hdr-container">
        <img src="data:image/avif;base64,{avif_b64}" 
             onclick="this.requestFullscreen()" 
             title="Click for fullscreen"/>
        <span class="fullscreen-hint">Click for fullscreen</span>
    </div>
    '''
    
    return html, status, output_path, heatmap_img, conf_stats


def create_ui():
    """Create the Gradio interface."""
    
    checkpoints = find_checkpoints()
    default_ckpt = checkpoints[0] if checkpoints else None
    checkpoint_dirs = find_checkpoint_dirs()
    default_dir = checkpoint_dirs[0] if checkpoint_dirs else None
    
    with gr.Blocks(title="X-Trans Demosaic") as demo:
        gr.Markdown("# X-Trans Demosaicing")
        
        with gr.Tabs():
            # Inference tab
            with gr.Tab("Inference"):
                gr.Markdown("Upload a Fujifilm RAF file. Output is HDR AVIF (HLG).")
                
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
                        
                        overlap_slider = gr.Slider(
                            minimum=0, maximum=264, step=24, value=48,
                            label="Tile Overlap",
                            info="Higher = more tiles per pixel, slower but better confidence map",
                        )

                        refresh_btn = gr.Button("🔄 Refresh Checkpoints", size="sm")
                        process_btn = gr.Button("Process", variant="primary")
                        
                        status_text = gr.Textbox(label="Status", interactive=False)
                        output_file = gr.File(label="Download AVIF")
                    
                    with gr.Column(scale=2):
                        output_html = gr.HTML(label="HDR Output")
                        with gr.Accordion("Tile Confidence Map", open=False):
                            confidence_img = gr.Image(label="Tile Disagreement (brighter = more uncertainty)")
                            confidence_stats = gr.Textbox(label="Stats", interactive=False)
                
                def refresh_checkpoints():
                    return gr.Dropdown(choices=find_checkpoints())
                
                refresh_btn.click(refresh_checkpoints, outputs=checkpoint_dropdown)
                
                process_btn.click(
                    run_inference,
                    inputs=[raf_input, checkpoint_dropdown, overlap_slider],
                    outputs=[output_html, status_text, output_file, confidence_img, confidence_stats],
                )
            
            # Training history tab
            with gr.Tab("Training History"):
                gr.Markdown("View training progress for checkpoint directories.")
                
                with gr.Row():
                    history_dir_dropdown = gr.Dropdown(
                        choices=checkpoint_dirs,
                        value=default_dir,
                        label="Checkpoint Directory",
                    )
                    refresh_history_btn = gr.Button("🔄 Refresh", size="sm")
                
                history_status = gr.Textbox(label="Status", interactive=False)
                history_plot = gr.Plot(label="Training History")
                
                def refresh_history_dirs():
                    dirs = find_checkpoint_dirs()
                    return gr.Dropdown(choices=dirs, value=dirs[0] if dirs else None)
                
                refresh_history_btn.click(refresh_history_dirs, outputs=history_dir_dropdown)
                
                history_dir_dropdown.change(
                    plot_training_history,
                    inputs=history_dir_dropdown,
                    outputs=[history_plot, history_status],
                )
                
                # Auto-load on tab open
                demo.load(
                    plot_training_history,
                    inputs=history_dir_dropdown,
                    outputs=[history_plot, history_status],
                )
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
