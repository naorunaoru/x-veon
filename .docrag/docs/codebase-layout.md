---
title: "Project Structure and Codebase Layout"
tags: [overview, structure]
scope: ["*.py", "src/**", "web/**", ".github/**"]
generated: 2026-03-22
commit: 347c8dd
---

## Context

X-veon is a neural network demosaicing system for Bayer and X-Trans camera sensors. The project has two major subsystems: a PyTorch training and inference pipeline (root-level Python files) and a browser-based RAW development application (the `web/` directory). The model architecture is a U-Net that takes raw CFA mosaic data and outputs full-color RGB images. Training data is synthesized from real RAW photographs via traditional demosaicing, downscaling, and re-mosaicing. The web application runs inference in-browser using ONNX Runtime WebGPU, and includes a full RAW processing pipeline with tone mapping, color grading, highlight reconstruction, and HDR export. A Gradio-based local UI (`ui.py`) provides an alternative inference interface for development and testing without the browser pipeline.

The codebase follows a flat-file layout for the Python training/inference components and a structured TypeScript/Rust application layout for the web frontend. There is a `src/` directory containing legacy copies of several modules; these files are outdated and scheduled for removal. All active Python development uses the root-level files exclusively.

## Pattern / Approach

### Top-Level Directory Structure

```
x-veon/
├── *.py                  # Active Python modules (model, training, dataset, inference, utilities)
├── environment.yml       # Conda environment specification for Python dependencies
├── web/                  # Browser-based RAW development app (React + WebGPU + WASM)
│   ├── src/              # TypeScript application source
│   ├── scripts/          # Build-time utilities (e.g. convert-lensfun-db.ts)
│   ├── wasm/             # Rust WASM crates (rawloader, demosaic, encoder)
│   └── public/           # Static assets including ONNX model files
├── src/                  # LEGACY — outdated copies, do not use
├── tests/                # Reference implementations and test fixtures
├── LICENSES/             # License texts (GPL-3.0-or-later, MIT, CC-BY-4.0)
├── .github/workflows/    # CI/CD (GitHub Pages deployment)
└── .docrag/              # Architecture documentation metadata
```

### Root-Level Python Files by Role

The root-level Python files are the active codebase for all model-related work. They are organized by function:

**Model Definition**

- `model.py` — Defines `XTransUNet`, the encoder-decoder U-Net architecture. Four encoder levels with `ConvBlock`, `DownBlock`, and `UpBlock` modules. Channel widths scale from `base_width` (default 64) through `base_width * 16` (1024) at the bottleneck. 5-channel input: CFA + R/G/B position masks + clip ratio. Uses GroupNorm (group=1, i.e. LayerNorm-style) instead of BatchNorm. PixelShuffle upsampling in `UpBlock`. The forward pass implements a residual CFA skip: the raw mosaic value is broadcast to all three output channels as a baseline, and the network learns color correction deltas. Accepts a `cfa_period` parameter (default 2); when > 2, adds sinusoidal positional encoding channels (sin/cos for row and column phase) and uses a 7x7 stem kernel. Also exports `count_parameters()` for model summary.

**Training Pipeline**

- `train.py` — Unified training script with two modes: `train` (initial, L1-focused) and `finetune` (MS-SSIM + gradient emphasis). Uses a `TrainConfig` dataclass as the single source of truth for all training parameters, with `TrainConfig.base()` and `TrainConfig.finetune()` presets. Handles the full training loop: data loading with train/val split at the image level (not the patch level) to prevent leakage, AMP mixed precision via `torch.amp`, cosine annealing LR schedule, checkpoint management (`best.pt` and `latest.pt` every 10 epochs), per-epoch history JSON logging, and a `TrainingDashboard` for live metrics. Imports `checkpoint_registry` for tracking checkpoints and `dashboard` for visualization. All loss weights, augmentation parameters, and model configuration are controlled via CLI arguments or the `TrainConfig` dataclass.

- `losses.py` — Composite loss function system built around the `DemosaicLoss` class. Components: L1 or Huber loss for pixel accuracy (drives PSNR), `SobelGradientLoss` for edge preservation, `MSSSIM` (multi-scale structural similarity from Wang et al. 2003, 5 scales), `ChromaLoss` for penalizing high-frequency false color artifacts in YCbCr space, `ZipperLoss` for penalizing spurious 2nd-order oscillations via Laplacian, and `ColorBiasLoss` for penalizing systematic DC color shift. Per-channel normalization mode addresses the green-channel sample count imbalance inherent in both X-Trans (20/36 green) and Bayer (2/4 green) CFAs. Supports `recon_only` mode which computes L1/Huber only on pixels under reconstruction (not sampled by CFA), with a configurable `known_pixel_weight` penalty to prevent drift at sampled positions. The `DemosaicLoss.base()` and `DemosaicLoss.finetune()` class methods are convenience constructors; actual training runs typically override individual weights via CLI or `TrainConfig`.

**Dataset Building**

- `build_dataset.py` — Converts RAW camera files (RAF, CR2, CR3, NEF, ARW, DNG, and others readable by rawpy) into training data. For each RAW file: demosaic using DHT (X-Trans) or AHD (Bayer), black-level subtract, normalize by white-minus-black range without clipping, downscale 4x via area averaging, and save as float32 `.npy` with a companion `_meta.json` sidecar containing camera white balance, range max, and CFA type. Accepts classification JSON files from the ranking scripts to select which images to process. Supports multiprocessing via `Pool`.

- `dataset.py` — PyTorch `Dataset` implementations for training. `LinearDataset` loads pre-computed `.npy` files and generates training patches on the fly: random crop aligned to CFA period, synthetic re-mosaicing through the CFA pattern, CFA channel mask generation, random flips, additive Gaussian noise, white balance perturbation in log space, exposure augmentation toward clipping, and optional OLPF (anti-aliasing filter) blur simulation. `create_mixed_dataset()` combines real image patches with synthetic torture patterns via `ConcatDataset` at a configurable ratio.

- `torture_v2.py` — Procedural generator for synthetic worst-case test patterns. Produces 4x-supersampled gradient stripes, Siemens stars, Nyquist grids, chromatic edges, Julia set fractals, Perlin noise, and other patterns that stress demosaicing at or near the Nyquist frequency of the CFA. These are mixed into training data at a low fraction (typically 5%) to improve robustness against moiré and zipper artifacts.

- `classify_raf.py` — Ranks RAF files by high-frequency high-amplitude content using Sobel gradient magnitude weighted by pixel intensity. Outputs a sorted JSON classification file used by `build_dataset.py` to select training images.

- `classify_hf_ha.py` — Similar ranking pipeline with a slightly different metric formulation (gradient times luminance at the 99th percentile). Outputs `hf_ha_ranking.json`.

- `backfill_metadata.py` — Retroactively generates `_meta.json` sidecar files for `.npy` datasets that were built before metadata extraction was added. Reads RAF headers via rawpy without re-demosaicing.

**Inference**

- `infer_hdr.py` — HDR inference pipeline with tile blending. BT.2020 color space support, linear-ramp tile blending to eliminate seams, EXIF-based rotation, and HDR AVIF output. Defines `apply_color_correction()` which implements the dcraw-style camera-to-output matrix computation (build forward sRGB-to-camera, row-normalize, invert). This module is imported by `ui.py` for its `process_raw()` and `save_hdr_avif()` functions.

- `highlight_recovery.py` — Python implementation of the opposed-channel highlight inpainting algorithm with dilated mask chrominance. Mirrors the TypeScript implementation in `preprocessor.ts` and the GPU implementation in `postprocess-gpu.ts`.

- `highlight_recovery_rgb.py` — RGB-domain highlight recovery variant. Operates on demosaiced 3-channel images rather than CFA mosaic data.

- `export_onnx.py` — Exports `XTransUNet` checkpoints to ONNX format for browser deployment. Embeds training metadata (epoch, validation PSNR, parameter count, optimizer and scheduler config) as ONNX model properties. Supports FP16 conversion via `onnxconverter_common.float16`. The exported `.onnx` files are served from `web/public/` for the browser application.

**UI**

- `ui.py` — Gradio web interface for local inference. Provides RAF file upload, checkpoint selection with model caching, tile overlap configuration, HDR AVIF export, confidence heatmap visualization, and training history chart display. Uses `infer_hdr.process_raw()` as its inference backend. Actively used during development for rapid iteration on model quality.

**Dataset Management and Visualization**

- `checkpoint_registry.py` — Checkpoint tracking utilities: `update_registry()` records checkpoints with metrics, `promote_to_stable()` marks a checkpoint as production-ready. Uses a JSON registry file.

- `dashboard.py` — `TrainingDashboard` for live training visualization. Provides `EpochData` for structured per-epoch metrics and real-time chart display.

- `preview_dataset.py` — Visualize training dataset patches and augmentations for debugging data pipeline issues.

- `profile_dataset.py` — Profile dataset loading performance: timing, I/O bottlenecks, and worker utilization.

- `prune_dataset.py` — Remove low-quality or redundant samples from training datasets.

- `stress.py` — Stress testing utility for the inference pipeline.

- `test_clip_ratio.py` — Test and visualize the clip ratio computation used in the 5th input channel.

**Utilities and Debugging**

- `cfa.py` — CFA (Color Filter Array) pattern definitions and core utilities. Defines `XTRANS_PATTERN` (6x6), `BAYER_PATTERN` (2x2), and the `CFA_REGISTRY` dictionary. Provides `make_cfa_mask()`, `make_channel_masks()`, `mosaic()`, `detect_cfa_from_raw()`, `find_pattern_shift()`, `cfa_period()`, and `patch_alignment()`. This is the canonical source for all CFA-related operations and is imported throughout the training and inference code.

- `xtrans_pattern.py` — Backward-compatibility shim that re-exports `XTRANS_PATTERN`, `make_cfa_mask`, `make_channel_masks`, and `mosaic` from `cfa.py`. Exists so that older code importing from `xtrans_pattern` continues to work.

- `fuji_decompress.py` — Python reference implementation of the Fuji compressed RAF decompressor, based on RawSpeed's `FujiDecompressor.cpp`. Used for debugging and validating the Rust WASM decoder, not for production inference.

- `save_reference.py` — Saves rawpy-decoded output as ground truth `.npy` arrays for testing the Fuji decompressor against known-good values.

- `compare_output.py` — Compares decoder output against reference arrays for diagnosing CFA mapping and decompression issues.

- `examine_raf.py` — Parses RAF file structure: TIFF IFD entries, compressed raw data offsets, and Fuji-specific header fields.

- `debug_first_strip.py` — Low-level debugging tool for verifying strip data offset and byte layout in compressed RAF files.

- `check_cfa.py` — Prints the CFA pattern, color description, and raw geometry reported by rawpy for a given RAF file.

### Web Application (`web/`)

The web application is a React + TypeScript SPA built with Vite, using WebGPU for rendering and compute, ONNX Runtime Web for neural network inference, and Rust-compiled WASM for RAW decoding, traditional demosaicing, and image encoding.

**Build System**

The project uses Vite as the bundler with `vite-plugin-wasm` for WASM integration. Three Rust WASM crates are compiled via `wasm-pack` before the Vite build. The build sequence is: `wasm-pack build` for each crate (rawloader, demosaic, encoder), then `vite build`. The `web/package.json` defines this as the `build:wasm` and `build` scripts. Deployment to GitHub Pages is handled by `.github/workflows/deploy.yml`, which also checks out the external `naorunaoru/rawloader` repository as a path dependency for the WASM rawloader crate.

**Dependencies**

Runtime: React 19, Zustand for state management, Radix UI primitives (dialog, slider, select, scroll-area, progress), Tailwind CSS 4, `onnxruntime-web` for neural network inference, `lucide-react` for icons, `clsx` and `tailwind-merge` for class composition.

Dev: Vite 6, `vite-plugin-wasm`, `wasm-pack`, `@webgpu/types`.

**`web/src/pipeline/`** — The RAW processing pipeline, module by module:

- `raf-decoder.ts` and `raf-thumbnail.ts` — WASM-based RAW file decoding. Loads sensor data, metadata (white balance, black/white levels, XYZ-to-camera matrix, crop boundaries, DR gain), and embedded JPEG thumbnails.
- `preprocessor.ts` — CFA preprocessing: crop-to-visible, white-point calibration (`calibrateWhiteLevels`), normalize to [0,1] per CFA position, mirror-pad to canonical CFA alignment, tile generation (288x288 tiles with 24px overlap), channel mask building, and batch CFA extraction (`prefillBatchMasks`, `fillBatchCfa` with 5-channel layout including clip ratio). Opposed-channel highlight reconstruction still defined here but now also implemented on GPU in `postprocess-gpu.ts`.
- `highlight-segments.ts` — Second-stage highlight reconstruction adapted from darktable: superpixel planes at 1/3 resolution, flood-fill segmentation of clipped regions, morphological closing, pseudo-chrominance transfer.
- `demosaic.ts` — Demosaic algorithm selection and dispatch. Routes to neural-net (tiled ONNX), WebGPU compute (bilinear, DHT), or WASM worker pool (Markesteijn, AHD, PPG, MHC).
- `demosaic-gpu.ts` — WebGPU compute shader implementations of bilinear and DHT demosaicing algorithms.
- `demosaic-pool.ts` and `demosaic-worker.ts` — WASM worker pool for traditional demosaicing algorithms. Up to 8 workers via `navigator.hardwareConcurrency`, horizontal strip splitting with CFA-period-aware overlap, per-strip WASM invocation.
- `inference.ts` — ONNX Runtime model loading and per-tile neural network inference. Loads both `xtrans.onnx` and `bayer.onnx` models with WebGPU-first execution provider (WASM fallback). Exposes `runTile()` for single-patch inference.
- `postprocessor.ts` — CPU post-demosaic processing: `cropToHWC()` for CHW-to-HWC conversion with padding removal, `buildColorMatrix()` and `applyColorCorrection()` for camera-to-sRGB transform.
- `postprocess-gpu.ts` — GPU compute pipeline for post-demosaic processing. Runs white balance, inpaint-opposed highlight recovery (passes 1-5: refavg+clip detection, chrominance calibration, chroma reconstruction), and finalize (HL extension + color correction + DR gain) in a single command buffer.
- `tile-blend-gpu.ts` — GPU-resident tile blend pipeline: `createGpuNNPipeline()` creates a pipeline that extracts tiles from CFA, accumulates inference output with overlap blending, and finalizes to HWC format -- all on the GPU without CPU readback.
- `encoder.ts` and `encoder-worker.ts` — Image export via WASM. Supported formats: UHD JPEG (3-channel gain map for HDR), AVIF (HLG transfer), uncompressed 16-bit TIFF.
- `color-temperature.ts` — Color temperature and tint estimation from WB coefficients using McCamy's CCT approximation.
- `constants.ts` — Shared constants: supported RAW file extensions, CFA patterns, tile sizes (`PATCH_SIZE=288`, `OVERLAP=24`, `TILE_BATCH=32`), and color space matrices.
- `types.ts` — TypeScript interfaces for `RawImage`, `CfaInfo`, `TileGrid`, `ChannelMasks`, `ProcessingResultMeta`, and other pipeline data structures. Type aliases: `CfaType`, `DemosaicMethod`, `ModelSize` (`'S' | 'M' | 'L'`), `ExportFormat`, `LookPreset`.

**`web/src/gl/`** — WebGPU rendering and compute:

- `renderer.ts` — The `HdrRenderer` class: dual render pipelines (canvas display and float-texture export), uniform buffer layout (400 bytes / 100 floats), histogram compute shaders for both SDR and HDR content, histogram reduction, and histogram visualization.
- `shaders/opendrt.wgsl` — OpenDRT tone mapping shader. Implements the full tonescale, per-channel adjustments, and color grading in WGSL.
- `shaders/histogram.wgsl` and `shaders/histogram-hdr.wgsl` — Compute shaders for SDR and HDR histogram generation.
- `shaders/histogram-reduce.wgsl` — Compute shader for histogram bin reduction.
- `shaders/histogram-viz.wgsl` — Compute shader for histogram visualization rendering.
- `color-matrices.ts` — Standard color space conversion matrices (XYZ to sRGB, XYZ to BT.2020) and camera-to-output matrix computation.
- `opendrt-params.ts` — OpenDRT configuration type definitions (`OpenDrtConfig`, `PreProcessConfig`, `GradingConfig`), look presets (base, default, colorful, umbra, flat), tonescale presets, and tonescale derivation that precomputes shader constants.
- `hdr-display.ts` — HDR display capability detection, headroom probing, and dual-render export strategy (SDR Rec.709 base + HDR Rec.2020 gain map).

**`web/src/hooks/`** — React hooks:

- `useInit.ts` — Application initialization: WASM loading, ONNX model loading, WebGPU context creation, session restore from IndexedDB + OPFS.
- `useProcessFile.ts` — File processing orchestration: decoding, preprocessing, demosaicing, postprocessing, and result caching per demosaic method.
- `useExport.ts` — Export flow: dual-render to float textures, WASM encoding, download trigger.
- `usePanZoom.ts` — Canvas pan/zoom interaction handling.

**`web/src/components/`** — React UI components:

- `App.tsx` (at `web/src/App.tsx`) — Root component and layout shell.
- `Sidebar.tsx`, `Header.tsx` — Navigation and branding.
- `FileList.tsx`, `FileListItem.tsx` — File queue display with status indicators, thumbnails, and camera metadata.
- `DropZone.tsx` — Drag-and-drop file input.
- `SettingsPanel.tsx` — Demosaic method selection and processing options.
- `GradingPanel.tsx` — Color grading controls: exposure, white balance (temperature/tint), per-channel adjustments, sharpen, OpenDRT look presets.
- `OutputPanel.tsx`, `OutputCanvas.tsx` — Image preview with WebGPU-rendered canvas.
- `Histogram.tsx` — Live histogram display.
- `ExportDialog.tsx` — Export format selection and progress.
- `HdrPermissionDialog.tsx` — HDR display permission prompt.
- `components/ui/` — Radix-based primitives: `button.tsx`, `slider.tsx`, `select.tsx`, `dialog.tsx`, `scroll-area.tsx`, `progress.tsx`, `switch.tsx`.

**`web/src/lib/`** — Shared utilities:

- `opfs-storage.ts` — Origin Private File System storage for RAW files and thumbnails. Two OPFS directories: `raw/` and `thumbnails/`.
- `idb-storage.ts` — IndexedDB persistence for file metadata, grading settings, and global preferences. Debounced writes for slider changes.
- `hwc-handoff.ts` — GPU buffer handoff mechanism between the processing pipeline and the display renderer, enabling zero-copy buffer sharing.
- `lensfun.ts` — LensFun lens database integration: `matchLens()` matches camera/lens metadata against the LensFun database to provide lens correction profiles (distortion, TCA, vignetting).
- `utils.ts` — General utilities (class name merging via `clsx` + `tailwind-merge`).

**`web/src/store.ts`** — Zustand store. Manages file queue (`QueuedFile` with status, result, lens profile, per-file grading overrides and pre-processing overrides), global settings (model size, demosaic method, export format, clip mask), and initialization state.

**`web/wasm/`** — Three Rust WASM crates:

- `wasm/rawloader/` — RAW file decoder. Path-depends on the external `naorunaoru/rawloader` repository (symlinked during CI). Exposes CFA data, metadata, and EXIF information to JavaScript.
- `wasm/demosaic/` — Traditional demosaicing algorithms compiled to WASM (Markesteijn for X-Trans, AHD/PPG/MHC for Bayer). Called by the worker pool.
- `wasm/encoder/` — Image encoding: UHDR JPEG with gain maps (`encode_uhdr.rs`), AVIF with HLG transfer (`encode_avif.rs`), 16-bit TIFF (`encode_tiff.rs`), standard JPEG (`encode_jpeg.rs`). Also handles EXIF embedding (`exif.rs`), orientation-aware rotation (`rotation.rs`), and transfer function conversion (`transfer.rs`). Orchestrated by `pipeline.rs`.

**`web/scripts/`** — Build-time utilities:

- `convert-lensfun-db.ts` — Converts the LensFun XML lens database into a compact JSON format for browser use.

**`web/public/`** — Served static assets. Contains the exported ONNX model files (`xtrans.onnx`, `bayer.onnx`, plus `.meta.json` sidecars), comparison page (`comparison.html`), and comparison images.

### Tests Directory

`tests/` contains reference implementations and test fixtures, not automated test suites:

- `tests/opendrt/` — OpenDRT reference: `opendrt_reference.py` (Python port of the transform), `opendrt_art.ctl` (Jed Smith's OpenDRT CTL with ART parameter annotations by agriggio), `opendrt_test_vectors.json` (golden vectors for cross-validation of the WGSL shader against the reference).

- `tests/hlrecon/` — Highlight reconstruction tests: `test_segbased.c` (C reference of the darktable segmentation algorithm), `test_ts.mjs` (JavaScript test runner), `compare.py` and `compare_cfa.mjs` (comparison utilities), `extract_fixture.py` (test data extraction).

### CI/CD

`.github/workflows/deploy.yml` deploys the web app to GitHub Pages on every push to `main`. The workflow checks out the main repo with LFS, checks out the external rawloader repository, symlinks it for the Cargo path dependency, installs Rust (stable, `wasm32-unknown-unknown` target) and Node 20, runs `npm ci`, builds three WASM crates via `wasm-pack`, runs `vite build`, and uploads `web/dist/` as a Pages artifact.

### Data and Checkpoint Conventions

Training data lives in `data/` (gitignored). `.npy` files contain float32 ground truth images; each has a companion `_meta.json` sidecar with camera white balance, range max, and sensor type. Checkpoints are stored in `checkpoints*/` directories (gitignored), containing `best.pt`, `latest.pt`, `config.json` (training arguments), and `history.json` (per-epoch metrics). Model weights (`.pt`, `.pth`, `.ckpt`, `.safetensors`) are excluded from git. ONNX exports for browser deployment are committed to `web/public/`.

## Rationale

The flat-file layout for Python modules avoids unnecessary packaging overhead for what is primarily a research and training codebase. Each file has a single, well-defined responsibility, and the module graph is shallow: `train.py` imports `model.py`, `dataset.py`, and `losses.py`; `dataset.py` imports `cfa.py` and `losses.py`; inference scripts import `model.py` and `cfa.py`. There is no `__init__.py` and no Python package structure because there is no need for one — all scripts run from the repository root.

The web application uses a more conventional structure because it is a production-deployed SPA with dozens of interdependent modules. The `pipeline/` directory isolates the RAW processing chain from the UI components, making it possible to reason about the image processing flow independently. The `gl/` directory groups all WebGPU concerns (renderer, shaders, display management). The `hooks/` directory contains React-specific orchestration that bridges store state to pipeline operations.

The three-crate WASM split (rawloader, demosaic, encoder) reflects the different lifecycle and dependency graphs of each concern. The rawloader depends on an external Rust library with its own release cycle. The demosaic crate contains CPU-bound algorithms that run in a worker pool. The encoder has heavy codec dependencies (JPEG, AVIF/rav1e, TIFF) that would bloat the other crates if combined.

The `DemosaicLoss.base()` and `DemosaicLoss.finetune()` class methods exist as convenience constructors that encode empirically-derived starting points. They do not represent the only valid configurations. Production training runs override individual loss weights via CLI arguments in `train.py` based on ongoing experimentation.

## Key Files

| File | Purpose |
|------|---------|
| `model.py` | `XTransUNet` definition with residual CFA skip |
| `train.py` | Training loop, CLI argument parsing, checkpoint management |
| `losses.py` | `DemosaicLoss` composite: L1/Huber, MS-SSIM, Sobel gradient, chroma, zipper, color bias |
| `dataset.py` | `LinearDataset` with CFA-aware augmentation, `create_mixed_dataset()` |
| `build_dataset.py` | RAW-to-npy conversion with DHT/AHD demosaic and 4x downscale |
| `cfa.py` | CFA pattern definitions, masking, mosaicing, phase detection |
| `infer_hdr.py` | HDR inference with tile blending and BT.2020 color correction |
| `highlight_recovery.py` | Python opposed-channel highlight inpainting (CFA domain) |
| `highlight_recovery_rgb.py` | Python RGB-domain highlight recovery variant |
| `checkpoint_registry.py` | Checkpoint tracking and promotion utilities |
| `export_onnx.py` | PyTorch-to-ONNX export with FP16 and metadata embedding |
| `ui.py` | Gradio local inference UI |
| `torture_v2.py` | Synthetic worst-case pattern generator for training augmentation |
| `web/src/pipeline/inference.ts` | Browser-side ONNX model loading and per-tile inference |
| `web/src/pipeline/preprocessor.ts` | CFA preprocessing, WP calibration, tile generation, batch CFA extraction |
| `web/src/pipeline/postprocess-gpu.ts` | GPU post-demosaic: WB, highlight recovery, CC, DR gain |
| `web/src/pipeline/tile-blend-gpu.ts` | GPU-resident tile blend pipeline for neural-net path |
| `web/src/lib/hwc-handoff.ts` | GPU buffer handoff between pipeline and renderer |
| `web/src/lib/lensfun.ts` | LensFun lens database matching |
| `web/src/pipeline/demosaic.ts` | Demosaic algorithm dispatch (neural/GPU/WASM) |
| `web/src/gl/renderer.ts` | WebGPU `HdrRenderer` with dual SDR/HDR pipelines |
| `web/src/gl/shaders/opendrt.wgsl` | OpenDRT tone mapping WGSL shader |
| `web/src/store.ts` | Zustand state: file queue, grading params, initialization |
| `web/wasm/rawloader/` | Rust WASM RAW file decoder |
| `web/wasm/encoder/` | Rust WASM image encoder (UHDR JPEG, AVIF, TIFF) |
| `.github/workflows/deploy.yml` | GitHub Pages deployment: WASM build + Vite build |

## Antipatterns

**Do not import from `src/`.** The `src/` directory contains `src/model.py`, `src/losses.py`, `src/xtrans_pattern.py`, `src/datasets/dataset_v4.py`, and `src/datasets/build_dataset_v4.py`. These are legacy copies that predate the current root-level files. They are out of date, may contain bugs that have been fixed in the root versions, and will be deleted. All imports should reference the root-level modules: `from model import XTransUNet`, `from cfa import CFA_REGISTRY`, `from losses import DemosaicLoss`, etc.

**Do not treat loss presets as fixed configurations.** `DemosaicLoss.base()` and `DemosaicLoss.finetune()` are convenience constructors that provide reasonable starting weights. They do not represent canonical or optimized configurations. Actual training invocations override individual weights (`--l1-weight`, `--msssim-weight`, `--gradient-weight`, `--chroma-weight`, `--color-bias-weight`) based on ongoing experimentation. Do not assume that calling `.base()` or `.finetune()` without overrides produces the best results.

**Do not confuse `xtrans_pattern.py` with `cfa.py`.** The file `xtrans_pattern.py` is a backward-compatibility shim that re-exports a subset of `cfa.py`. New code should import directly from `cfa.py`, which is the authoritative source for all CFA pattern definitions (X-Trans and Bayer), mask generation, pattern detection, and alignment utilities.

**Do not commit model weights to git.** The `.gitignore` excludes `.pt`, `.pth`, `.ckpt`, and `.safetensors` files, as well as `checkpoints*/` directories. ONNX exports for the web app are the exception — `web/public/*.onnx` files are tracked because they are required for the deployed application.

**Do not add Python package structure.** There is no `__init__.py`, `setup.py`, or `pyproject.toml` for the Python code. The training scripts are designed to run directly from the repository root (`python train.py ...`). Adding packaging infrastructure would create maintenance burden without benefit for this use case.
