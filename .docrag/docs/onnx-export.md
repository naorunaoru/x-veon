---
title: ONNX Export Pipeline
tags: [model, onnx, export, deployment]
scope: export_onnx.py
generated: 2026-03-22
commit: 347c8dd
---

## Context

### Bridging PyTorch Training to Browser Inference

The x-veon project trains demosaicing models in PyTorch and deploys them for browser-based inference via ONNX Runtime Web. The `export_onnx.py` script is the sole bridge between these two environments. It converts a PyTorch checkpoint (`.pt`) into a self-contained ONNX file (`.onnx`) that the web application loads at startup and runs tile-by-tile against raw camera sensor data.

The exported model is the `XTransUNet` architecture documented in [U-Net Demosaicing Model Architecture](unet-model-architecture.md). The same export pipeline handles both X-Trans and Bayer variants of the model. Rather than exporting individual checkpoints manually, the script iterates a `checkpoint_registry.json` manifest with filtering options (`--cfa-type`, `--base-width`, `--variant`, `--status`) and batch-exports all matching checkpoints to `web/public/checkpoints/`. Each exported model is named by its registry label (e.g., `xtrans_w64_base.onnx`, `bayer_w32_hl.onnx`).

The browser consumption side lives in `web/src/pipeline/inference.ts`, which fetches `checkpoints/models.json` at startup, resolves the best model key for each CFA type and size, and creates ONNX Runtime sessions on demand. The export script must produce files that satisfy three constraints simultaneously: (1) ONNX Runtime Web can load them without external data dependencies, (2) the numerical output matches the PyTorch original within acceptable tolerances, and (3) the file size is small enough for practical browser delivery.

## Pattern / Approach

### Export Flow Overview

The export pipeline proceeds in five sequential stages per checkpoint:

1. **Checkpoint loading and model reconstruction** -- Load the `.pt` checkpoint, read `base_width` and `cfa_period` from it, instantiate `XTransUNet(base_width=..., cfa_period=...)`, and load the state dict.
2. **ONNX tracing** -- Run `torch.onnx.export()` with a 5-channel dummy input tensor to trace the model into an ONNX graph.
3. **Self-contained file consolidation** -- Reload the ONNX file and re-save it with all weights inlined (no external data files).
4. **Optional FP16 conversion** -- Convert internal weights to float16 while preserving float32 I/O types.
5. **Verification** -- Optionally compare ONNX outputs against PyTorch outputs using PSNR thresholds.

Metadata is no longer embedded in ONNX `metadata_props` or a per-model sidecar `.meta.json`. Instead, the `_extract_metadata()` return value (epoch, base_width, param_count, size_mb, dtype) plus per-model training metrics from the registry (train_psnr, val_psnr, train_loss, val_loss) are aggregated into a single `models.json` manifest in the output directory.

### Registry-Based Batch Export

In the default mode (no `--checkpoint` flag), `main()` loads `checkpoint_registry.json` and iterates it with `_iter_registry()`, which yields `(label, checkpoint_path, registry_entry)` tuples. The registry is organized as a nested dict: `registry[cfa_type][base_width][variant][status][slot]`. CLI filters (`--cfa-type`, `--base-width`, `--variant`, `--status`) narrow the iteration; unspecified filters match all entries. The `--slot` flag selects `best` or `latest` checkpoints (default: `best`).

**SHA256-based caching:** Before exporting each checkpoint, the script computes the SHA256 hash of the source `.pt` file and compares it against the `source_sha256` stored in the existing `models.json` manifest. If the hash matches and the `.onnx` file still exists on disk, the export is skipped and the old manifest entry is reused. The `--force` flag bypasses this check and re-exports unconditionally.

The model label (e.g., `xtrans_w64_base`, `bayer_w32_hl`) is constructed as `{cfa_type}_w{base_width}_{variant}` and used as both the ONNX filename stem and the manifest key.

### Stage 1: Checkpoint Loading

```python
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
bw = base_width or ckpt.get("base_width", 64)
cp = _ckpt_cfa_period(ckpt)
model = XTransUNet(base_width=bw, cfa_period=cp)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
```

The `base_width` is read from the checkpoint metadata, defaulting to 64 for backward compatibility with older checkpoints. The `cfa_period` is derived from the checkpoint's `cfa_type` field via the CFA registry (`_ckpt_cfa_period` calls `cfa_period(CFA_REGISTRY[cfa_type])`). The model is loaded with `strict=False` to accommodate architecture evolution. The model is set to eval mode to freeze BatchNorm running statistics before tracing.

### Stage 2: ONNX Tracing

```python
dummy = torch.randn(1, 5, patch_size, patch_size)
torch.onnx.export(
    model, dummy, output_path,
    opset_version=opset,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

The dummy input shape is `(1, 5, patch_size, patch_size)` -- batch of 1, 5-channel input (CFA values + position masks + clip ratio), with spatial dimensions matching the inference tile size. Additional positional encoding channels (sin/cos for row/column phase, when `cfa_period > 2`) are generated internally by the model. The default `patch_size` is 288, matching the `PATCH_SIZE` constant in `web/src/pipeline/constants.ts`. The default opset version is 18.

Input and output are given fixed names (`"input"` and `"output"`) that the browser consumer references directly. Dynamic axes are specified for the batch dimension only (`{0: "batch"}`), which allows batched inference while keeping spatial dimensions fixed. The model is fully convolutional and can accept any spatial dimensions divisible by 16, but the ONNX graph is traced with the fixed patch size that the browser consumer will use.

### Stage 3: Self-Contained File Consolidation

```python
onnx_model = onnx.load(output_path, load_external_data=True)
onnx.checker.check_model(onnx_model)

# ... (fp16 conversion happens here if --fp16) ...

ext_data = Path(output_path + ".data")
if ext_data.exists():
    ext_data.unlink()

onnx.save(onnx_model, output_path, save_as_external_data=False)
```

PyTorch's ONNX exporter may produce a separate `.data` file for large weight tensors. This step reloads the model, runs the ONNX checker for structural validity, and re-saves with `save_as_external_data=False` to inline all weights into the single `.onnx` file. Any external data file is explicitly deleted.

This consolidation is required because ONNX Runtime Web cannot load external data files -- it expects a single self-contained `.onnx` blob fetched via URL.

### Stage 4: FP16 Conversion

When `--fp16` is passed:

```python
onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
```

The `onnxconverter_common.float16` converter transforms all internal weights and intermediate computations from float32 to float16. The `keep_io_types=True` flag preserves float32 for the model's input and output tensors. This means:

- The browser sends float32 input and receives float32 output (no type conversion needed in JavaScript).
- Internal computation and weight storage use float16, roughly halving model file size.
- The browser-side inference code in `inference.ts` always constructs `ort.Tensor('float32', ...)` regardless of the model's internal precision.

The verification step attempts to detect float16 models by inspecting `sess.get_inputs()[0].type`. However, because `keep_io_types=True` preserves float32 I/O, the input type is always `tensor(float)` even for FP16-converted models. This means the FP16 threshold (35 dB) is never selected — all models are verified against the stricter FP32 threshold (60 dB).

### Metadata: `models.json` Manifest

Instead of embedding metadata in ONNX `metadata_props` or per-model sidecar files, the export script writes a single `models.json` manifest in the output directory. Each key is the model label (e.g., `xtrans_w64_base`), and each value contains:

| Key | Source |
|-----|--------|
| `epoch` | `ckpt["epoch"]` |
| `base_width` | `ckpt["base_width"]` |
| `param_count` | Computed from state dict tensor sizes |
| `size_mb` | ONNX file size on disk |
| `dtype` | `"float16"` or `"float32"` |
| `file` | ONNX filename (e.g., `xtrans_w64_base.onnx`) |
| `source_sha256` | SHA256 of the source `.pt` checkpoint |
| `train_psnr` | From the checkpoint registry |
| `val_psnr` | From the checkpoint registry |
| `train_loss` | From the checkpoint registry |
| `val_loss` | From the checkpoint registry |

The browser fetches this manifest at startup:

```typescript
// web/src/pipeline/inference.ts
manifest = await fetchManifest(`${CHECKPOINTS_DIR}/models.json`);
```

The browser then resolves per-CFA-type model keys from the manifest (via `resolveModelKey`) and loads sessions on demand.

### Verification

The `--verify` flag triggers a numerical comparison between PyTorch and ONNX outputs on identical random input:

1. Run the same random 5-channel tensor through both the PyTorch model and the ONNX Runtime session.
2. Compute max absolute difference, mean absolute difference, and PSNR between the two outputs.
3. Apply a precision-dependent PSNR threshold:
   - **FP32 model**: PSNR must exceed **60 dB** (near bit-exact).
   - **FP16 model**: PSNR must exceed **35 dB** (allows for half-precision quantization noise).
4. Print PASS or WARN based on the threshold.

The verification uses `onnxruntime` (CPU provider) locally, not the browser's WebGPU provider. This catches export-time errors (graph tracing mistakes, operator incompatibilities) but does not test the browser execution path.

The FP16 threshold of 35 dB is intentionally conservative. In practice, FP16 exports of the deployed models achieve well above this threshold. The purpose of the check is to catch catastrophic failures (e.g., a NaN-producing operator conversion), not to guarantee perceptual equivalence.

### Output Files

A batch export invocation produces:

| File | Description |
|------|-------------|
| `<label>.onnx` | Self-contained ONNX model with all weights inlined (e.g., `xtrans_w64_base.onnx`, `bayer_w32_hl.onnx`). Tracked in git via LFS. |
| `models.json` | Aggregated manifest with per-model metadata for all exported checkpoints. |

The default output directory is `web/public/checkpoints/`. Model labels follow the convention `{cfa_type}_w{base_width}_{variant}` derived from the registry structure. The `inference.ts` consumer fetches `checkpoints/models.json` at startup, resolves the best model key for each CFA type and requested size, and loads sessions on demand via `getOrLoadSession`.

A legacy single-checkpoint mode is still available via `--checkpoint` / `--output` flags, which skips the registry and exports a single file (default output: `web/public/model.onnx`).

### CLI Interface

```
python export_onnx.py [options]
```

**Registry-based batch export (default mode):**

| Flag | Default | Description |
|------|---------|-------------|
| `--cfa-type` | all | Filter by sensor type (`xtrans`, `bayer`) |
| `--base-width` | all | Filter by base width (`16`, `32`, `64`) |
| `--variant` | all | Filter by variant name |
| `--status` | prefer stable | Filter by status (`stable`, `beta`) |
| `--slot` | `best` | Which checkpoint slot to export (`best`, `latest`) |
| `--output-dir` | `web/public/checkpoints` | Output directory for batch export |
| `--force` | off | Re-export even if source checkpoint is unchanged |

**Legacy single-checkpoint mode:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (none) | Export a single checkpoint (skips registry) |
| `--output` | `web/public/model.onnx` | Output path (only with `--checkpoint`) |

**Shared export options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--patch-size` | `288` | Spatial dimension of the traced input tensor |
| `--opset` | `18` | ONNX opset version |
| `--fp16` | off | Convert weights to float16 (keeps I/O as float32) |
| `--verify` | off | Run numerical comparison after export |

### Typical Export Commands

Export all registered checkpoints with FP16 and verification:

```bash
python export_onnx.py --fp16 --verify
```

Export only X-Trans models at base_width=32:

```bash
python export_onnx.py --cfa-type xtrans --base-width 32 --fp16 --verify
```

Force re-export of all Bayer models regardless of SHA cache:

```bash
python export_onnx.py --cfa-type bayer --fp16 --force
```

Legacy single-checkpoint export:

```bash
python export_onnx.py \
  --checkpoint checkpoints_xtrans/best.pt \
  --output web/public/model.onnx \
  --fp16 --verify
```

### Browser Consumption Path

The exported ONNX files are consumed through the following chain:

1. **`useInit.ts`** calls `initModels(size)` during application startup, in parallel with WASM loading and GPU initialization.
2. **`inference.ts` / `initModels()`** fetches `checkpoints/models.json`, then for each CFA type resolves the best model key for the requested size via `resolveModelKey(cfaType, width)` (preferring `_hl` variants over `_base`). It creates ONNX Runtime sessions on demand via `getOrLoadSession`, trying the WebGPU execution provider first, falling back to multi-threaded WASM, then single-threaded WASM.
3. **`useProcessFile.ts`** calls `runTile(cfaType, input, PATCH_SIZE)` for each 288x288 tile during neural-net demosaicing. The CFA type (`'xtrans'` or `'bayer'`) selects which loaded session to use.
4. **`inference.ts` / `runTile()`** constructs an `ort.Tensor('float32', ...)`, runs the session with `{ input: tensor }`, and returns the `output` tensor data as a `Float32Array`.

The tensor name contract (`"input"` / `"output"`) is hardcoded on both sides -- in `export_onnx.py` via `input_names` / `output_names` and in `inference.ts` via the run call and result access. Model size switching at runtime is supported via `switchModelSize()`.

## Rationale

### Why a Single Self-Contained File

ONNX Runtime Web fetches models via URL. If weights were stored in a separate `.data` file (as PyTorch's exporter sometimes produces for models above ~2GB), the browser runtime would need to locate and fetch that file separately. ONNX Runtime Web's `InferenceSession.create()` accepts a single URL and has no mechanism for resolving external data references. Consolidating everything into one file eliminates this problem and simplifies deployment -- the model is a single static asset served from `web/public/checkpoints/`.

### Why FP16

The deployed X-Trans model has ~7.8M parameters (base_width=32) and the Bayer model has ~1.9M parameters (base_width=16). At float32, the X-Trans model is roughly 30 MB; FP16 halves this to approximately 15 MB. For a browser-delivered application, this size reduction meaningfully improves initial load time, especially on mobile connections.

The `keep_io_types=True` flag is critical: it preserves float32 at the model boundaries so the browser code does not need to handle float16 tensor creation or output interpretation. JavaScript's `Float32Array` is the natural transport type for ONNX Runtime Web, and float16 typed arrays are not widely supported. The FP16 conversion is purely an internal optimization -- weights and intermediate activations use half precision, but the API surface remains float32.

The quality cost of FP16 is minimal for this model. BatchNorm layers are the most sensitive to precision reduction, but the `onnxconverter_common` converter handles them correctly. The verification step's 35 dB PSNR threshold provides a safety net.

### Why a Single `models.json` Manifest

Rather than embedding metadata in ONNX `metadata_props` (which ONNX Runtime Web's JavaScript API does not expose) or writing per-model sidecar `.meta.json` files, the export script writes a single `models.json` manifest that maps model labels to metadata. This simplifies both the export pipeline and browser consumption: the browser makes one fetch to discover all available models and their metadata, rather than N+1 fetches (one per model sidecar plus the model files themselves). The manifest also enables model size switching and variant resolution on the client side without any filesystem probing.

### Why the Verification Approach

The verification step compares ONNX Runtime CPU output against PyTorch output, not the browser's WebGPU output. This is intentional: the export script runs in the training environment where `onnxruntime` (Python) is available but a browser is not. The verification catches the most likely failure modes -- graph tracing errors, unsupported operators, and FP16 conversion problems -- without requiring a browser test harness.

PSNR is used rather than simple max-diff because the signal magnitude varies across random inputs. A max-diff of 1e-3 might be acceptable for a signal in [0, 1] but catastrophic for a signal in [0, 1e-6]. PSNR normalizes by the signal range, giving a scale-invariant quality metric. The dual thresholds (60 dB for FP32, 35 dB for FP16) reflect the fundamentally different precision guarantees of the two formats.

### Why Opset 18

Opset 18 is the default because it is the most recent opset that is reliably supported across ONNX Runtime Web's WebGPU and WASM backends at the time of deployment. The `XTransUNet` architecture uses only standard operators (Conv, BatchNorm, ReLU, MaxPool, ConvTranspose, Add) that are available in much older opsets, but using a recent opset ensures the broadest operator coverage if the model architecture evolves.

## Key Files

| File | Role |
|------|------|
| `export_onnx.py` | The complete export pipeline: registry iteration, tracing, consolidation, FP16, SHA caching, manifest generation, verification. |
| `checkpoint_registry.json` | Registry of all checkpoints organized by CFA type, base width, variant, and status. Built by `checkpoint_registry.py`. |
| `checkpoint_registry.py` | Builds and updates `checkpoint_registry.json`. Defines `REGISTRY_FILENAME`. |
| `cfa.py` | `CFA_REGISTRY` and `cfa_period()` used by `_ckpt_cfa_period()` to derive the model's CFA period from checkpoint metadata. |
| `web/src/pipeline/inference.ts` | Browser-side ONNX Runtime session management: fetches `models.json`, resolves model keys, creates sessions on demand, runs tile inference. |
| `web/src/pipeline/constants.ts` | Defines `PATCH_SIZE = 288` and `OVERLAP = 24` used by both export (as default `--patch-size`) and browser inference. |
| `web/src/hooks/useInit.ts` | Calls `initModels(size)` during app startup to load ONNX sessions for the requested model size. |
| `web/src/hooks/useProcessFile.ts` | Calls `runTile()` per tile during neural-net demosaicing. |
| `web/public/checkpoints/` | Output directory for exported ONNX files and `models.json` manifest. |
| `.gitattributes` | Configures `*.onnx` files for Git LFS tracking. |
| `model.py` | `XTransUNet` class definition (now takes `in_channels=5` and `cfa_period` parameter). See [U-Net Demosaicing Model Architecture](unet-model-architecture.md). |

## Antipatterns

### Do Not Change Input/Output Tensor Names

The tensor names `"input"` and `"output"` are a contract between `export_onnx.py` and `inference.ts`. The browser code references these names as literal strings in `session.run({ input: tensor })` and `results.output.data`. Changing the names in the export script without updating the browser code (or vice versa) will cause a silent runtime failure where `session.run()` throws because the input name does not match, or `results.output` is `undefined`.

### Do Not Add Dynamic Axes for Spatial Dimensions

The export currently uses `dynamic_axes` for the batch dimension only (`{0: "batch"}`). Do not add dynamic axes for the spatial dimensions (height, width). ONNX Runtime Web's WebGPU backend optimizes for fixed tensor shapes, and introducing spatial dynamic axes can prevent shape-based kernel selection and fusion optimizations. The browser always sends tiles at the fixed `PATCH_SIZE` (288x288), so spatial dynamic axes provide no benefit and may reduce performance.

### Do Not Remove the Self-Contained File Consolidation

The reload-and-resave step (`onnx.load` followed by `onnx.save` with `save_as_external_data=False`) may appear redundant. It is not. PyTorch's ONNX exporter can produce external data files for models whose protobuf representation exceeds 2 GB, and even for smaller models the consolidation guarantees a clean single-file output. Removing this step risks producing models that work locally (where the `.data` file is co-located) but fail in the browser (where only the `.onnx` URL is fetched).

### Do Not Skip Verification After FP16 Conversion

FP16 conversion can introduce numerical issues that are not obvious from file size or structure alone. Operators with large dynamic range (e.g., BatchNorm scale factors near zero) may produce NaN or Inf values after conversion. Always run `--verify` when exporting with `--fp16`. If PSNR drops below the 35 dB threshold, investigate which layers are causing precision loss before deploying the model.

### Do Not Bloat `models.json` with Large Data

The `_extract_metadata()` function deliberately keeps the per-model metadata small (epoch, base_width, param_count, size_mb, dtype). The `models.json` manifest is fetched by the browser at startup; bloating it with large blobs (e.g., full training history, per-epoch metrics) would increase initial load time. Keep per-model entries compact and use separate files for any large ancillary data.
