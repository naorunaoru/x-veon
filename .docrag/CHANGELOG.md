# Documentation Changelog

## 2026-03-22 — 2e6b96c..347c8dd

The inference pipeline moved from CPU-resident processing with OPFS caching to a fully GPU-resident pipeline with in-memory buffer handoff. Neural-net demosaicing now runs entirely on GPU: tile extraction, batched ONNX inference, weighted blending, and post-processing (white balance, highlight recovery, color correction) all execute as WebGPU compute passes without CPU readback. This eliminated the OPFS HWC storage layer — reprocessing is now faster than decompressing cached results.

The U-Net architecture was reworked: BatchNorm replaced with GroupNorm(1) (per-sample LayerNorm equivalent), MaxPool with strided convolution, ConvTranspose with PixelShuffle upsampling. The model now takes 5 input channels (CFA value + 3 position masks + clip ratio) instead of 4, and accepts a `cfa_period` parameter that adds sinusoidal positional encoding for non-Bayer patterns (X-Trans). The residual skip changed from broadcasting the CFA value to all 3 channels to per-channel masking (`cfa * masks`), so each output channel's baseline is the actual sampled value at that position (zero elsewhere).

The web app gained multi-model support (S/M/L sizes backed by different `base_width` checkpoints), a checkpoint registry and manifest system for organized model management, LensFun lens profile matching from EXIF metadata, GPU-accelerated histogram visualization, and an expanded color grading UI with new OpenDRT parameters and preset system.

On the training side: a `TrainConfig` dataclass replaced ad-hoc CLI argument handling, the dataset pipeline gained clip ratio computation, bright-spot augmentation, downscale augmentation, and a streaming `PatchCacheDataset` for large datasets. New loss components include `ZipperLoss` and reconstruction-only mode. Python-side highlight recovery scripts (`highlight_recovery.py`, `highlight_recovery_rgb.py`) implement the darktable-compatible two-pass algorithm for use during inference.

### Changes

- **unet-model-architecture**: BatchNorm→GroupNorm, MaxPool→strided conv, ConvTranspose→PixelShuffle, 5-ch input with clip ratio, `cfa_period` param with positional encoding, per-channel residual baseline
- **loss-functions**: New ZipperLoss, recon_only mode, Huber loss switch, per_channel_norm option
- **training-loop**: TrainConfig dataclass, registry integration, PatchCacheDataset, new augmentation CLI args
- **training-augmentations**: Clip ratio channel output, bright-spot augmentation, downscale augmentation, PatchCacheDataset/ImageGroupedSampler
- **onnx-export**: Registry-based batch export, models.json manifest, multi-model naming convention, SHA256 caching
- **tiled-inference-blending**: GPU-resident pipeline via tile-blend-gpu.ts, batched inference (TILE_BATCH=32), OVERLAP reduced 48→24
- **cfa-preprocessing-web**: Clip ratio computation, calibrated white levels, 5-channel tile input assembly
- **color-correction-pipeline**: Moved to GPU postprocess-gpu.ts (6-pass compute pipeline), CPU fallback path retained
- **opfs-pixel-storage**: HWC OPFS storage removed, replaced by single-slot GPU buffer handoff (hwc-handoff.ts)
- **webgpu-renderer**: Shared device pattern with ORT, GPU histogram pipeline (histogram-reduce.wgsl, histogram-viz.wgsl)
- **opendrt-tone-mapping**: New params (hc_r, cwp, cwp_rng, pt_hdr), PreProcessConfig type, expanded presets
- **web-processing-pipeline-flow**: Entirely new GPU-resident neural-net path, batched tile loop, GPU postprocess
- **pipeline-type-contracts**: New ModelSize, LookPreset types; ProcessingResultMeta gains modelSize; new store fields
- **zustand-state-management**: New fields: lensProfile, modelSize, showClipMask, preProcessOverrides; removed cached result restore
- **session-persistence**: HWC OPFS layer removed; session restore now re-processes instead of loading cached pixels
- **demosaic-algorithm-dispatch**: Neural-net path now GPU-resident with batching; traditional path unchanged
- **gradio-inference-ui**: Highlight reconstruction mode selector (cfa/rgb), checkpoint dropdown with caching, confidence heatmap
- **modification-recipes**: Multiple recipes affected by GPU pipeline, new types, changed extension points
- **codebase-layout**: New files: highlight_recovery.py, highlight_recovery_rgb.py, checkpoint_registry.py, dashboard.py, lensfun.ts, postprocess-gpu.ts, tile-blend-gpu.ts, hwc-handoff.ts, histogram shaders
- **highlight-reconstruction-opposed**: Python-side implementation now exists alongside TypeScript

### New docs needed

- **gpu-resident-nn-pipeline**: tile-blend-gpu.ts + postprocess-gpu.ts — the 4-stage GPU extract/accumulate/finalize/crop pipeline and 6-pass GPU postprocessing
- **checkpoint-registry**: checkpoint_registry.py + models.json manifest + multi-model inference selection
- **lensfun-integration**: lensfun.ts — lens profile matching from EXIF, string normalization, Jaccard scoring
- **python-highlight-recovery**: highlight_recovery.py + highlight_recovery_rgb.py — two-pass darktable-compatible HL reconstruction
