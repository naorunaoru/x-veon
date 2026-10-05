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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from export_onnx import load_model as load_checkpoint_model
from infer_hdr import apply_exif_rotation, process_raw, save_hdr_avif


# Global state
_model = None
_model_path = None
_device = None
_UI_STATE_FILE = Path(".ui_state.json")


def load_ui_state() -> dict:
    if _UI_STATE_FILE.exists():
        try:
            return json.loads(_UI_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_ui_state(key: str, value):
    state = load_ui_state()
    state[key] = value
    _UI_STATE_FILE.write_text(json.dumps(state))


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_checkpoints():
    """Find all available checkpoints."""
    patterns = [
        "checkpoints/**/best.pt",
        "checkpoints/**/latest.pt",
    ]
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob(pattern, recursive=True))
    return sorted(set(checkpoints), reverse=True)


def find_checkpoint_dirs():
    """Find all checkpoint directories with history.json."""
    dirs = sorted(glob("checkpoints/**/", recursive=True))
    return [d.rstrip("/") for d in dirs if Path(d, "history.json").exists()]


def load_history(checkpoint_dir: str) -> list[dict]:
    """Load training history from checkpoint directory."""
    history_path = Path(checkpoint_dir) / "history.json"
    if not history_path.exists():
        return []
    with open(history_path) as f:
        history = json.load(f)
    import math
    return [h for h in history if not math.isnan(h.get("val_loss", 0))]


def plot_training_history(checkpoint_dir: str) -> tuple:
    """Generate interactive training history plots using Plotly."""
    history = load_history(checkpoint_dir)
    if not history:
        return None, "No history found"

    epochs = [h["epoch"] for h in history]
    train_psnr = [h["train_psnr"] for h in history]
    val_psnr = [h["val_psnr"] for h in history]

    # Load config for title
    config_path = Path(checkpoint_dir) / "config.json"
    config_str = ""
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        parts = []
        if cfg.get("color_bias_weight"): parts.append(f"color_bias={cfg['color_bias_weight']}")
        if cfg.get("stages"): parts.append(f"{cfg['stages']} stages")
        config_str = ", ".join(parts)

    title = f"{checkpoint_dir}  —  {config_str}" if config_str else checkpoint_dir

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("PSNR", "Train Components", "Val Components"),
        horizontal_spacing=0.06,
    )

    # --- PSNR plot ---
    best_idx = int(np.argmax(val_psnr))
    fig.add_trace(go.Scatter(
        x=epochs, y=train_psnr, name="Train PSNR",
        mode="lines", opacity=0.5,
        hovertemplate="Epoch %{x}<br>Train PSNR: %{y:.2f} dB<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=val_psnr, name="Val PSNR",
        mode="lines", opacity=0.8,
        hovertemplate="Epoch %{x}<br>Val PSNR: %{y:.2f} dB<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[epochs[best_idx]], y=[val_psnr[best_idx]],
        name=f"Best: {val_psnr[best_idx]:.2f} dB (ep {epochs[best_idx]})",
        mode="markers+text",
        marker=dict(size=10, color="green", symbol="star"),
        text=[f"{val_psnr[best_idx]:.2f} dB"],
        textposition="top center",
        hovertemplate="Best: %{y:.2f} dB at epoch %{x}<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text="PSNR (dB)", row=1, col=1)

    # --- Component plots (weighted contributions) ---
    COMP_COLORS = {
        "l1": "#1f77b4", "l1_recon": "#4169e1", "l1_known": "#6495ed",
        "huber": "#1f77b4", "huber_recon": "#4169e1", "huber_known": "#6495ed",
        "color_bias": "#8c564b",
    }
    # Map component names to their config weight keys.
    # l1_recon/l1_known are sub-components of l1 — use l1_weight for them.
    def _get_weight(comp):
        if cfg.get("recon_only"):
            if comp == "l1_recon":
                w = cfg.get("l1_weight")
            elif comp == "l1_known":
                w = cfg.get("known_pixel_weight")
            else:
                w = cfg.get(f"{comp}_weight")
        elif comp in ("l1_recon", "l1_known"):
            w = cfg.get("l1_weight")
        else:
            w = cfg.get(f"{comp}_weight")
        return w if w is not None else 1.0

    def add_components(history, key, col, show_legend):
        if key not in history[0]:
            return
        all_comps = dict.fromkeys(comp for h in history for comp in h[key])
        for comp in all_comps:
            if comp == "total":
                continue
            raw = [h[key].get(comp, 0) for h in history]
            if not any(v > 0 for v in raw):
                continue
            w = _get_weight(comp)
            weighted = [v * w for v in raw]
            label = f"{comp} (×{w:g})" if w != 1.0 else comp
            # Show both weighted and raw in hover
            custom = [f"raw: {r:.4e}" for r in raw]
            fig.add_trace(go.Scatter(
                x=epochs, y=weighted, name=label,
                mode="lines", opacity=0.8,
                legendgroup=comp, showlegend=show_legend,
                line=dict(color=COMP_COLORS.get(comp)),
                customdata=custom,
                hovertemplate=f"{comp}<br>Epoch %{{x}}<br>Weighted: %{{y:.4e}}<br>%{{customdata}}<extra></extra>",
            ), row=1, col=col)
        fig.update_yaxes(type="log", title_text="Weighted Loss", row=1, col=col)

    add_components(history, "train_components", col=2, show_legend=True)
    add_components(history, "val_components", col=3, show_legend=False)
    # Share y-axis range between train and val component plots
    fig.update_yaxes(matches="y2", row=1, col=3)

    # --- Layout ---
    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=400,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=20, t=40, b=20),
    )

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
    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=True)
    _model = load_checkpoint_model(ckpt, checkpoint_path).to(_device)
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
    hlrecon: str = "cfa",
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

    rgb_linear, meta = process_raw(raf_path, model, str(device), patch_size=patch_size, overlap=overlap,
                                    hlrecon=hlrecon)

    progress(0.9, desc="Encoding HDR AVIF...")

    output_path = tempfile.mktemp(suffix=".avif", prefix=f"{raf_name}_hdr_")
    save_hdr_avif(rgb_linear, output_path, 90,
                  exif_flip=meta.get("exif_flip", 0),
                  dr_gain=meta.get("dr_gain", 1.0))

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
    ui_state = load_ui_state()
    saved_ckpt = ui_state.get("checkpoint")
    default_ckpt = saved_ckpt if saved_ckpt in checkpoints else (checkpoints[0] if checkpoints else None)
    checkpoint_dirs = find_checkpoint_dirs()
    saved_dir = ui_state.get("history_dir")
    default_dir = saved_dir if saved_dir in checkpoint_dirs else (checkpoint_dirs[0] if checkpoint_dirs else None)
    
    with gr.Blocks(title="X-Trans Demosaic") as demo:
        gr.Markdown("# X-Trans Demosaicing")
        
        with gr.Tabs(selected="inference"):
            # Inference tab
            with gr.Tab("Inference", id="inference"):
                gr.Markdown("Upload a raw file (RAF, CR2, NEF, ARW, DNG, ...). Output is HDR AVIF (HLG).")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        raf_input = gr.File(
                            label="Raw File",
                            file_types=[".RAF", ".raf", ".CR2", ".cr2", ".CR3", ".cr3",
                                        ".NEF", ".nef", ".ARW", ".arw", ".DNG", ".dng"],
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

                        hlrecon_radio = gr.Radio(
                            choices=["cfa", "rgb"],
                            value="cfa",
                            label="Highlight Reconstruction",
                            info="cfa = pre-demosaic (darktable), rgb = post-demosaic",
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

                refresh_btn.click(refresh_checkpoints,
                                  outputs=[checkpoint_dropdown])

                checkpoint_dropdown.change(
                    lambda v: save_ui_state("checkpoint", v),
                    inputs=checkpoint_dropdown,
                )

                process_btn.click(
                    run_inference,
                    inputs=[raf_input, checkpoint_dropdown, overlap_slider, hlrecon_radio],
                    outputs=[output_html, status_text, output_file, confidence_img, confidence_stats],
                )
            
            # Training history tab
            with gr.Tab("Training History", id="training"):
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
                
                def on_history_dir_change(d):
                    save_ui_state("history_dir", d)
                    return plot_training_history(d)

                history_dir_dropdown.change(
                    on_history_dir_change,
                    inputs=history_dir_dropdown,
                    outputs=[history_plot, history_status],
                )

                # Live auto-refresh every 30 seconds
                timer = gr.Timer(30)
                timer.tick(
                    plot_training_history,
                    inputs=history_dir_dropdown,
                    outputs=[history_plot, history_status],
                )

                # Restore saved state + plot on every page load
                def on_load():
                    state = load_ui_state()
                    dirs = find_checkpoint_dirs()
                    ckpts = find_checkpoints()
                    saved_d = state.get("history_dir")
                    saved_c = state.get("checkpoint")
                    dir_val = saved_d if saved_d in dirs else (dirs[0] if dirs else None)
                    ckpt_val = saved_c if saved_c in ckpts else (ckpts[0] if ckpts else None)
                    fig, status = plot_training_history(dir_val) if dir_val else (None, "No history found")
                    return (
                        gr.Dropdown(choices=dirs, value=dir_val),
                        fig,
                        status,
                        gr.Dropdown(choices=ckpts, value=ckpt_val),
                    )

                demo.load(
                    on_load,
                    outputs=[history_dir_dropdown, history_plot, history_status,
                             checkpoint_dropdown],
                )
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
