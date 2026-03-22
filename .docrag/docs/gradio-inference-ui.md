---
title: Gradio Local Inference UI
tags: [ui, gradio, inference, python]
scope: ui.py
generated: 2026-03-22
commit: 347c8dd
---

## Context

### Local Inference and Model Evaluation Outside the Browser

The x-veon project includes a browser-based WebGPU demosaicing pipeline for real-time use, but model development requires a separate workflow: loading PyTorch checkpoints, running full-resolution inference on raw camera files, inspecting HDR output, and comparing training runs. These tasks cannot run in the WebGPU frontend because they depend on PyTorch, rawpy, and the full Python training ecosystem.

`ui.py` provides a Gradio web application that serves as the local counterpart to the browser app. It exposes two capabilities through a tabbed interface: single-image demosaicing inference with HDR AVIF output, and training history visualization from checkpoint directories. The UI runs on any machine with a GPU (CUDA or MPS) or on CPU, and can be shared externally via Gradio's tunnel feature.

The model loaded by this UI is `XTransUNet`, documented in [U-Net Demosaicing Model Architecture](unet-model-architecture.md). Despite the X-Trans name, the architecture is CFA-agnostic and the file upload widget accepts RAF, CR2, CR3, NEF, ARW, and DNG raw files.

## Pattern / Approach

### Gradio App Structure

The application uses `gr.Blocks` with a two-tab layout:

- **Inference tab.** Left column: file upload, checkpoint selector, tile overlap slider, highlight reconstruction mode radio (`cfa`/`rgb`), refresh and process buttons, status text, and a download link. Right column: the rendered HDR image displayed as inline HTML, with a collapsible accordion for the confidence heatmap and its statistics.
- **Training History tab.** A dropdown for selecting a checkpoint directory (any directory under `checkpoints/` that contains a `history.json`), a refresh button, a status line, and a matplotlib plot. A `gr.Timer(30)` auto-refreshes the plot every 30 seconds during active training.

The app is launched from `__main__` with `argparse` flags for `--port` (default 7860) and `--share` (enables Gradio's public URL tunnel). The server binds to `0.0.0.0` so it is reachable from other machines on the local network.

```
python ui.py [--port 7860] [--share]
```

### Raw File Upload and Processing

The `run_inference` function is the core handler wired to the "Process" button. It accepts four inputs from the Gradio components: the uploaded raw file path, the selected checkpoint string, the overlap integer, and the highlight reconstruction mode (`hlrecon`: `"cfa"` or `"rgb"`).

Processing delegates to `process_raw` from `infer_hdr.py`, which handles the full pipeline: rawpy decoding, black/white level normalization, white balance application at the CFA level, CFA pattern auto-detection and alignment, optional highlight recovery (pre-demosaic via `highlight_recovery.reconstruct_highlights` when `hlrecon="cfa"`, or post-demosaic via `highlight_recovery_rgb.reconstruct_highlights` when `hlrecon="rgb"`), tiled model inference with overlap blending, and crop back to original dimensions. The function returns linear RGB in camera space plus a metadata dictionary containing white balance multipliers, the XYZ-to-camera matrix, EXIF orientation, a confidence map (when overlap is nonzero), and the Fuji DR gain.

Progress is reported through Gradio's `gr.Progress` callback at four stages: model loading (10%), demosaicing start (20%), AVIF encoding (90%), and completion (100%). The `track_tqdm=True` parameter on the progress object also forwards any tqdm bars from the inference loop.

### Model Checkpoint Selection with Caching

`find_checkpoints` scans for `.pt` files matching two glob patterns:

- `checkpoints/**/best.pt` -- best validation PSNR for each training run (recursive)
- `checkpoints/**/latest.pt` -- most recent periodic save (recursive)

The results are deduplicated and sorted in reverse lexicographic order so the most recent runs appear first. A "Refresh Checkpoints" button re-scans the filesystem and updates the dropdown without restarting the app, which is useful during active training when new checkpoints appear.

Model caching uses a simple global-variable strategy rather than an LRU cache with a fixed capacity. Three module-level globals (`_model`, `_model_path`, `_device`) store the currently loaded model. When `load_model` is called, it compares the requested checkpoint path against `_model_path`; if they match and `_model` is not `None`, it returns the cached model immediately. If the path differs, it loads the new checkpoint, replacing the previous one. This means exactly one model is resident in memory at a time -- there is no eviction policy because GPU memory constraints make holding multiple full models impractical.

The checkpoint dictionary is expected to contain `model` (state dict), optionally `base_width` (defaults to 64 if absent), `cfa_type` (defaults to `"xtrans"`), `epoch`, and `best_val_psnr`. Both `base_width` and `cfa_period` (derived from `cfa_type` via `_ckpt_cfa_period()`) are passed to `XTransUNet` so the UI can load checkpoints from training runs with different channel widths and CFA types. The state dict is loaded with `strict=False` to accommodate architecture evolution. The checkpoint is loaded with `weights_only=True`. The model is set to `eval()` mode and the device is auto-detected via `get_device()`, which prefers MPS (Apple Silicon), then CUDA, then CPU.

### Tile Overlap Configuration

The overlap slider ranges from 0 to 264 in steps of 24, with a default of 48. The patch size is fixed at 288 pixels. Stride is computed as `patch_size - overlap`, so the default configuration produces a stride of 240.

When overlap is 0, tiles are placed without any blending -- each pixel is computed by exactly one tile. When overlap is nonzero, `process_raw` constructs a 1D linear ramp weight that goes from 0 to 1 across the overlap region at the start of each tile and from 1 to 0 at the end. The 2D blend weight is the outer product of two such 1D ramps, creating a smooth bilinear falloff in all four corners and edges.

The blending accumulates weighted tile outputs and weighted squared outputs into separate accumulators. After all tiles are processed, the weighted mean is computed by dividing by the accumulated weights. The variance is derived from `E[x^2] - E[x]^2` (the weighted sum-of-squares minus the squared weighted mean). This variance drives the confidence heatmap. Higher overlap means more tiles contribute to each pixel, which improves blending smoothness and produces a more meaningful variance signal, at the cost of proportionally more model forward passes.

### HDR AVIF Output with Fullscreen

After `process_raw` returns linear RGB, the UI calls `save_hdr_avif` to produce a 10-bit AVIF file encoded with HLG (Hybrid Log-Gamma) transfer characteristics. The `dr_gain` from metadata is passed through: when greater than 1.0 (indicating Fuji's deliberate underexposure for dynamic range extension), the linear RGB is multiplied by `dr_gain` before encoding to compensate. The CICP signaling is `9/18/9`: BT.2020 primaries, HLG transfer function, BT.2020 non-constant luminance matrix. The encoding pipeline converts linear RGB through camera-to-BT.2020 color correction, applies the HLG OETF, quantizes to 16-bit PNG as an intermediate, and shells out to `avifenc` for the final compression.

The AVIF is not served as a file download alone. Because Gradio's `gr.Image` component cannot display HDR AVIF natively, the UI base64-encodes the AVIF and embeds it in an `<img>` tag inside a `gr.HTML` component. The HTML includes inline CSS and a click handler:

- Clicking the image calls `this.requestFullscreen()`, enabling full-window HDR viewing on capable displays.
- A translucent "Click for fullscreen" hint is positioned at the bottom-right corner.
- In fullscreen mode, the image uses `object-fit: contain` with a black background.

The status line reports the image dimensions, the checkpoint name (directory/filename), and the count of HDR pixels (those with linear values exceeding 1.0 before tone mapping).

A `gr.File` download link is also provided so the user can save the AVIF to disk.

### Confidence Heatmap from Tile Variance

When overlap is nonzero, `process_raw` returns a `confidence_map` in the metadata dictionary. This is the per-pixel root-mean-square deviation (RMSD) across the three color channels, computed from the tile blending variance: `sqrt(mean(var, axis=channels))`. It measures how much different overlapping tiles disagree about each pixel's value.

`make_confidence_heatmap` converts this scalar map to a displayable image:

1. Applies EXIF rotation to match the output image orientation.
2. Computes the 99th percentile (`p99`) as the normalization ceiling to prevent extreme outliers from washing out the visualization.
3. Normalizes to `[0, 1]` by dividing by `p99` and clamping.
4. Maps through matplotlib's `inferno` colormap (dark-to-bright-yellow), producing a uint8 RGB image.

The stats string reports four values: mean variance, max variance, the p99 threshold, and the percentage of pixels exceeding p99. High-variance regions typically correspond to fine texture, strong edges, or areas where the CFA pattern creates aliasing -- all locations where the demosaicing problem is most ambiguous. This gives the developer a spatial quality diagnostic beyond aggregate PSNR.

The heatmap is displayed inside a collapsible `gr.Accordion` labeled "Tile Confidence Map", defaulting to closed since it is a diagnostic tool rather than the primary output.

### Training History Visualization

The Training History tab reads `history.json` files that `train.py` writes after every epoch. `load_history` filters out entries with NaN `val_loss` values. Each valid entry contains epoch number, train/val loss, train/val PSNR, per-component loss breakdowns (`train_components`, `val_components`), learning rate, and wall-clock time.

`plot_training_history` generates a 1x3 matplotlib figure:

- **Left panel: PSNR over epochs.** Shows training PSNR and validation PSNR (both at low alpha). An annotation arrow marks the best validation PSNR with the epoch number and dB value.
- **Center panel: Training loss components.** Plots individual loss terms (L1/Huber, MS-SSIM, gradient, chroma, zipper, color bias) on a log scale, each with a dedicated color from `COMP_COLORS`. MS-SSIM is converted from similarity to loss (`1 - value`) for consistent visualization. Only components with nonzero values are shown.
- **Right panel: Validation loss components.** Same breakdown as the center panel but for the validation set.

The plot title is derived from the checkpoint directory name plus key hyperparameters extracted from `config.json` in the same directory: MS-SSIM weight, per-channel normalization flag, color bias weight, torture fraction, and white balance augmentation. This makes it easy to visually distinguish training runs.

The status line shows the current epoch, latest validation PSNR, and best validation PSNR with its epoch number.

The `on_load` handler (wired to `demo.load`) restores both the checkpoint dropdown and history directory dropdown from persisted state on every page load, and immediately renders the training history plot for the restored directory.

A `gr.Timer(30)` auto-refreshes the plot every 30 seconds, and a refresh button re-scans for checkpoint directories that may have appeared during an active training session.

### UI State Persistence

User selections (checkpoint, history directory) are persisted to a `.ui_state.json` file via `save_ui_state(key, value)` and restored on page load via `load_ui_state()`. The checkpoint dropdown's `change` event and the history directory dropdown's `change` event both call `save_ui_state` so selections survive browser refreshes and server restarts. Persistence is best-effort: `load_ui_state` catches `JSONDecodeError` and `OSError` and falls back to an empty dict.

## Rationale

### Why Gradio Alongside the Web App

The browser-based WebGPU pipeline and the Gradio UI serve fundamentally different audiences and workflows:

- **The web app** targets end users performing real-time demosaicing with an exported ONNX model. It runs entirely client-side with no Python dependency.
- **The Gradio UI** targets model developers who need to evaluate PyTorch checkpoints against full-resolution raw files, inspect per-pixel diagnostics, compare training run histories, and iterate on model improvements. It requires the full training environment (PyTorch, rawpy, matplotlib).

Gradio was chosen over alternatives (Streamlit, Panel, custom Flask app) for several reasons:

- **Component library matches the use case.** File upload, sliders, dropdowns, matplotlib plot embedding, and HTML injection are all built-in components. No custom frontend code is needed.
- **Minimal boilerplate.** The entire UI is defined in a single `create_ui` function under 100 lines. Adding a new control or output is a few lines of Gradio API calls.
- **Progress tracking.** `gr.Progress` integrates with tqdm, providing real-time feedback during the minutes-long inference process for a full-resolution raw file.
- **`--share` for remote access.** During training on a remote GPU server, the share tunnel provides immediate browser access without configuring port forwarding or reverse proxies.

### Device Auto-Detection

The `get_device` function checks MPS before CUDA. This ordering reflects the development setup: Apple Silicon laptops are used for UI development and quick evaluations, while dedicated CUDA machines handle training. The function has no configuration knob because the correct device is nearly always the only accelerator present on the machine.

## Key Files

| File | Role |
|------|------|
| `ui.py` | Gradio application: layout, event handlers, model caching, heatmap rendering, history plotting, UI state persistence |
| `infer_hdr.py` | `process_raw` (tiled inference with overlap blending, highlight recovery, and variance), `save_hdr_avif` (HLG encoding with DR gain compensation via avifenc), `apply_exif_rotation` |
| `highlight_recovery.py` | `reconstruct_highlights` -- pre-demosaic (CFA-level) highlight recovery, used when `hlrecon="cfa"` |
| `highlight_recovery_rgb.py` | `reconstruct_highlights` -- post-demosaic (RGB-level) highlight recovery, used when `hlrecon="rgb"` |
| `model.py` | `XTransUNet` model class (now takes `cfa_period` parameter) loaded by the UI (see [unet-model-architecture](unet-model-architecture.md)) |
| `cfa.py` | `CFA_REGISTRY` and `cfa_period()` used to derive `cfa_period` from checkpoint's `cfa_type` |
| `train.py` | Writes `history.json` and `config.json` consumed by the Training History tab |
| `.ui_state.json` | Persisted UI selections (checkpoint, history directory) |
| `checkpoints/**/best.pt` | Checkpoint files discovered by `find_checkpoints` (recursive glob) |
| `checkpoints/**/history.json` | Per-epoch training metrics consumed by `plot_training_history` |

## Antipatterns

### Do Not Add an LRU Cache for Multiple Models

The current single-model caching strategy (`_model` / `_model_path` globals) is intentional. An LRU cache holding multiple loaded models would consume GPU memory proportional to the cache size. A single `XTransUNet` with `base_width=64` occupies significant VRAM; caching two or three would risk OOM on consumer GPUs. If A/B comparison between checkpoints is needed, the correct approach is to reload on each switch (which takes under a second for the model sizes involved), not to hold multiple models resident.

### Do Not Replace the HTML Image Embed with gr.Image

The base64-encoded `<img>` tag inside `gr.HTML` exists because `gr.Image` processes uploads through PIL, which strips HDR metadata and cannot display AVIF with HLG transfer characteristics. The HTML approach preserves the full HDR signal path to the browser's image decoder. Replacing it with `gr.Image` would silently clip to SDR.

### Do Not Set Overlap to Maximum by Default

While higher overlap produces smoother blending and richer confidence maps, it increases inference time quadratically. At overlap=264 with patch_size=288, the stride is only 24 pixels, meaning each pixel is covered by roughly 144 overlapping tiles. The default of 48 (stride=240) keeps single-image inference under a minute on a modern GPU while still providing meaningful tile disagreement data. Reserve high overlap values for targeted quality investigations, not routine use.
