---
title: U-Net Demosaicing Model Architecture
tags: [model, neural-net, architecture, unet]
scope: model.py
generated: 2026-03-22
commit: 347c8dd
---

## Context

### The Demosaicing Problem

Digital camera sensors capture light through a Color Filter Array (CFA) — a repeating mosaic of colored filters where each photosite records only one color channel. The demosaicing task is to reconstruct a full three-channel RGB image from this single-channel mosaic input. This is an ill-posed problem: at every pixel, two of the three color values are missing and must be inferred from the spatial neighborhood.

Traditional demosaicing algorithms use hand-crafted interpolation with edge-detection heuristics. These work adequately for the simple 2x2 Bayer pattern but struggle with the more complex 6x6 X-Trans pattern, where the larger repeat period and irregular green distribution create artifacts that directional interpolation cannot resolve cleanly.

### Why a U-Net

The U-Net encoder-decoder architecture is well suited to demosaicing for several reasons:

- **Multi-scale feature extraction.** The encoder's strided-convolution stages build a hierarchy of spatial scales, allowing the network to combine fine local texture with broader color context. Demosaicing requires both: local gradients determine edge direction, while larger neighborhoods resolve ambiguous color transitions.
- **Skip connections preserve spatial detail.** The decoder receives high-resolution features directly from the encoder at each level, preventing the loss of sharp edges that a pure encoder-decoder would suffer from spatial downsampling.
- **Receptive field scales naturally.** Four levels of 2x downsampling yield a theoretical receptive field that easily covers multiple CFA repeat periods (the X-Trans 6x6 pattern being the most demanding), giving the network enough context to learn the full color interpolation geometry.
- **Fully convolutional.** The architecture processes arbitrary spatial dimensions (subject to alignment constraints), enabling patch-based training and variable-size inference without architectural changes.

## Pattern / Approach

### XTransUNet Class

The model is implemented as `XTransUNet` in `model.py`. Despite the name (a historical artifact from when the project targeted only Fujifilm X-Trans sensors), the architecture is fully CFA-agnostic. It handles both X-Trans and Bayer patterns identically — all CFA-specific information is encoded in the input tensor's mask and clip-ratio channels (and, when `cfa_period > 2`, sin/cos positional encodings generated in the forward pass), not in the network structure. See the [CFA Pattern System](cfa-pattern-system.md) documentation for details on how the 5-channel input is constructed.

### Constructor Parameters

```python
XTransUNet(in_channels=5, out_channels=3, base_width=64, cfa_period=2)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_channels` | 5 | CFA value (1 ch) + R/G/B binary position masks (3 ch) + clip ratio (1 ch). See [CFA Pattern System](cfa-pattern-system.md) for the 5-channel input format. |
| `out_channels` | 3 | RGB output. |
| `base_width` | 64 | Controls all channel widths as a multiplier. Encoder/decoder widths are `base_width * [1, 2, 4, 8]` and the bottleneck is `base_width * 16`. |
| `cfa_period` | 2 | Repeat period of the CFA pattern (2 for Bayer, 6 for X-Trans). When `cfa_period > 2`, the model prepends 4 channels of sin/cos positional encoding to the input and uses a 7x7 stem kernel in `enc1`. |

The `base_width` parameter is the primary knob for trading model capacity against inference cost. It is saved in checkpoints and restored at load time via `ckpt.get("base_width", 64)`.

### Channel Width Scaling

All layer widths derive from `base_width` (abbreviated `w` in the code):

| Level | Encoder | Bottleneck | Decoder |
|-------|---------|------------|---------|
| 1 | w | | w |
| 2 | w*2 | | w*2 |
| 3 | w*4 | | w*4 |
| 4 | w*8 | | w*8 |
| 5 | | w*16 | |

Common configurations:

| base_width | Bottleneck Channels | Approximate Parameters | Use Case |
|------------|--------------------:|----------------------:|----------|
| 64 | 1024 | ~31M | Full training |
| 32 | 512 | ~7.8M | Deployed X-Trans model (web) |
| 16 | 256 | ~1.9M | Deployed Bayer model (web) |

### Building Blocks

#### ConvBlock

The fundamental unit is a double convolution: the first convolution uses a configurable kernel size (`first_kernel`, default 3), and the second is always 3x3. Each convolution is followed by `GroupNorm(1, out_ch)` (equivalent to LayerNorm over spatial dimensions) and ReLU. Bias is enabled (the default for `nn.Conv2d`).

```
Conv2d(first_kernel x first_kernel, pad=first_kernel//2) -> GroupNorm(1) -> ReLU
Conv2d(3x3, pad=1) -> GroupNorm(1) -> ReLU
```

The `first_kernel` parameter is used by `enc1` to apply a 7x7 stem convolution when `cfa_period > 2` (X-Trans), and defaults to 3 for all other blocks.

Every encoder level, decoder level, and the bottleneck use a `ConvBlock` as their core processing unit.

#### DownBlock

Downsamples by 2x using a learned strided convolution `Conv2d(in_ch, in_ch, 2, stride=2)`, then applies a `ConvBlock`. Used for encoder levels 2-4 and the bottleneck. The first encoder level (`enc1`) uses a bare `ConvBlock` with no pooling because it operates at full resolution.

#### UpBlock

Upsamples by 2x using a 1x1 convolution `Conv2d(in_ch, out_ch * 4, 1)` followed by `PixelShuffle(2)`, which rearranges the expanded channels into spatial dimensions. The upsampled tensor is concatenated with the corresponding encoder skip connection along the channel dimension, then processed by a `ConvBlock`. The `ConvBlock` input width is `out_ch * 2` to accommodate the skip concatenation.

### Architecture Diagram

```
Input (B, 5, H, W)  [+4 pos_enc channels when cfa_period > 2]
  |
  v
enc1: ConvBlock(5 -> w)  [or ConvBlock(9 -> w, 7x7 stem) when cfa_period > 2]                          ----skip1--->  dec1: UpBlock(w*2 -> w)
  |                                                                    |
  v                                                                    v
enc2: DownBlock(w -> w*2)                         ----skip2--->  dec2: UpBlock(w*4 -> w*2)
  |                                                                    |
  v                                                                    v
enc3: DownBlock(w*2 -> w*4)                       ----skip3--->  dec3: UpBlock(w*8 -> w*4)
  |                                                                    |
  v                                                                    v
enc4: DownBlock(w*4 -> w*8)                       ----skip4--->  dec4: UpBlock(w*16 -> w*8)
  |                                                                    ^
  v                                                                    |
bottleneck: DownBlock(w*8 -> w*16)  --->  (B, w*16, H/16, W/16)  -----+
```

After the decoder, a 1x1 convolution maps from `w` channels to 3 (RGB). The residual CFA skip is then added to produce the final output.

### Encoder Path

1. **Level 1** (`enc1`): `ConvBlock(in_channels + pos_channels, w, first_kernel=stem_kernel)` at full resolution `(H, W)`. For Bayer (`cfa_period=2`), this is `ConvBlock(5, w)` with a 3x3 stem. For X-Trans (`cfa_period > 2`), 4 sin/cos positional encoding channels are concatenated, giving `ConvBlock(9, w, first_kernel=7)`.
2. **Level 2** (`enc2`): `DownBlock(w, w*2)` at `(H/2, W/2)`.
3. **Level 3** (`enc3`): `DownBlock(w*2, w*4)` at `(H/4, W/4)`.
4. **Level 4** (`enc4`): `DownBlock(w*4, w*8)` at `(H/8, W/8)`.

### Bottleneck

`DownBlock(w*8, w*16)` at `(H/16, W/16)`. This is the deepest level with the largest channel count and smallest spatial extent. The 16x spatial reduction means each bottleneck neuron has access to a 16x16-pixel region even before accounting for the 3x3 convolution receptive fields.

### Decoder Path

Each decoder level receives two inputs: the upsampled features from the level below and the skip connection from the corresponding encoder level.

1. **Level 4** (`dec4`): `UpBlock(w*16, w*8)` — concatenates with `enc4` output, processes at `(H/8, W/8)`.
2. **Level 3** (`dec3`): `UpBlock(w*8, w*4)` — concatenates with `enc3` output, processes at `(H/4, W/4)`.
3. **Level 2** (`dec2`): `UpBlock(w*4, w*2)` — concatenates with `enc2` output, processes at `(H/2, W/2)`.
4. **Level 1** (`dec1`): `UpBlock(w*2, w)` — concatenates with `enc1` output, processes at `(H, W)`.

### Skip Connections

Each encoder level's output is passed directly to the corresponding decoder level via concatenation along the channel dimension. This is the standard U-Net skip connection pattern. The skip connections allow the decoder to access the encoder's high-resolution feature maps, which is critical for demosaicing because the output must preserve the spatial precision of the input — every pixel needs an accurate color estimate.

### Output Layer and Residual CFA Skip

The output layer is a 1x1 convolution: `Conv2d(w, 3, kernel_size=1)`. This projects the decoder's `w`-channel feature map down to 3 channels (RGB).

The model's most distinctive design feature is the **residual CFA skip**. Before the encoder processes the input, the forward method extracts channel 0 (the raw CFA mosaic value) and the R/G/B position masks (channels 1-3), then computes a per-channel baseline by multiplying them:

```python
cfa = x[:, 0:1]    # (B, 1, H, W)
masks = x[:, 1:4]  # (B, 3, H, W) — R, G, B position masks
baseline = cfa * masks  # (B, 3, H, W) — value only in its true channel
```

Each output channel's baseline is the actual sampled CFA value at positions where that color was recorded, and zero elsewhere. The final output is `baseline + out_conv(d1)` — the network's learned correction is *added* to the per-channel CFA baseline rather than producing absolute RGB values.

### Input and Output Format

**Input**: `(B, 5, H, W)` tensor.
- Channel 0: mosaiced CFA value (spatially dense, every pixel has a value)
- Channel 1: binary mask for red positions (1.0 where filter is red, 0.0 elsewhere)
- Channel 2: binary mask for green positions
- Channel 3: binary mask for blue positions
- Channel 4: clip ratio (how close each pixel is to sensor saturation)

When `cfa_period > 2`, the forward pass concatenates 4 additional sin/cos positional encoding channels (row sine, row cosine, column sine, column cosine, with angular frequency `2 * pi / cfa_period`), bringing the effective input to 9 channels.

This representation is constructed by the CFA pattern system. See [CFA Pattern System](cfa-pattern-system.md) for mask generation and the rationale for this 5-channel design over alternatives like 3-channel sparse packing.

**Output**: `(B, 3, H, W)` tensor — full-color RGB at the same spatial resolution as the input. Values are in the same linear intensity space as the input CFA data.

### Spatial Constraints

Input dimensions must be divisible by 16 (the total downsampling factor: 4 levels of 2x strided convolution = 2^4). Additionally, dimensions must be divisible by the CFA period for mask alignment. The combined constraint is `lcm(cfa_period, 16)` — which is 48 for X-Trans and 16 for Bayer. The `patch_alignment()` function in `cfa.py` computes this value. See [CFA Pattern System](cfa-pattern-system.md) for alignment details.

## Rationale

### Residual CFA Skip for Exposure Agnosticism

The residual CFA skip is the single most important architectural decision. By having the network learn color *deltas* rather than absolute values, the model becomes exposure-agnostic: scaling the input by any constant factor scales both the CFA baseline and the network's delta proportionally (assuming the network's learned function is approximately scale-equivariant, which GroupNorm encourages). This means a model trained on one exposure range generalizes to others without retraining.

Without the residual skip, the network would need to learn the identity mapping for pixels where the CFA already provides the correct color. With the skip, those pixels require near-zero output from the network, concentrating the model's capacity on the actual interpolation task.

The baseline is constructed by multiplying channel 0 (the raw CFA value) with the R/G/B position masks (channels 1-3). This places each pixel's recorded value into only its true color channel, with zeros in the other two. At positions where the CFA records green, the baseline has the correct green value and zeros for red and blue; the network only needs to predict the red and blue corrections. At red positions, the red channel has the correct value and the network corrects green and blue. This structure is explicit in the mask arithmetic rather than requiring the network to disambiguate a broadcast value.

### CFA-Agnostic Design

The `XTransUNet` architecture contains no CFA-specific logic. The same model class, with identical weights, processes both X-Trans and Bayer inputs. All CFA geometry is encoded in the input tensor's mask channels (channels 1-3), which are generated externally by the CFA pattern system. The model's `in_channels=5` and `cfa_period` are the only architectural surfaces that reflect the CFA representation.

This design means adding support for a new CFA pattern (e.g., quad-Bayer) requires no model changes — only a new pattern definition in `cfa.py` and an entry in `CFA_REGISTRY`. In practice, separate checkpoints are trained for different CFA types because the interpolation patterns differ, but the architecture and training pipeline are shared.

### Receptive Field Considerations

Demosaicing requires the network to "see" enough spatial context to resolve the CFA pattern. The minimum useful receptive field is at least one full CFA repeat period (6 pixels for X-Trans, 2 for Bayer), but larger receptive fields improve quality at edges and in textured regions.

The theoretical receptive field of this architecture is substantial. Each `ConvBlock` applies two 3x3 convolutions, contributing a 5x5 receptive field per block. With four encoder levels and four decoder levels (plus bottleneck), the effective receptive field spans well over 100 pixels — easily covering 2-3 X-Trans repeats (12-18 pixels) even in the conservative estimate. The bottleneck's 16x spatial reduction means each bottleneck feature aggregates information from a 16x16-pixel region before the convolutions further expand it.

For X-Trans, this is critical: the 6x6 pattern means a pixel's correct color depends on which of 36 possible positions it occupies, and the surrounding context must span enough of the pattern to resolve ambiguities. For Bayer (2x2 period), the receptive field is far more than sufficient.

### Configurable base_width

The `base_width` parameter allows scaling the model to match deployment constraints without changing the architecture. The default of 64 produces a ~31M parameter model suitable for GPU training. The deployed web models use `base_width=32` for X-Trans (~7.8M parameters) and `base_width=16` for Bayer (~1.9M parameters), reflecting the fact that Bayer's simpler 2x2 pattern requires less model capacity than X-Trans's 6x6 pattern.

The `base_width` is saved in training checkpoints and ONNX model metadata, ensuring that inference code always reconstructs the correct architecture. All consumers retrieve it via `ckpt.get("base_width", 64)`, defaulting to 64 for backward compatibility with older checkpoints that predate the configurable width feature.

## Key Files

| File | Role |
|------|------|
| `model.py` | Canonical model definition: `XTransUNet`, `ConvBlock`, `DownBlock`, `UpBlock`, `count_parameters()`. This is the active version with configurable `base_width` and `cfa_period`. |
| `train.py` | Training script; instantiates `XTransUNet(base_width=args.base_width)`, saves `base_width` in checkpoints. |
| `infer.py` | Inference script; loads `base_width` from checkpoint. |
| `infer_hdr.py` | HDR inference pipeline; loads `base_width` from checkpoint. |
| `export_onnx.py` | ONNX export; embeds `base_width` in model metadata for browser deployment. |
| `ui.py` | Gradio UI; loads `base_width` from checkpoint. |
| `cfa.py` | CFA pattern system that generates the 5-channel input tensor consumed by this model. See [CFA Pattern System](cfa-pattern-system.md). |

## Antipatterns

### Do Not Import from src/model.py

The file `src/model.py` is a **legacy version** of the model definition. It lacks both the configurable `base_width` parameter (all channel widths are hardcoded to the 64/128/256/512/1024 progression) and the residual CFA skip connection (it returns `self.out_conv(d1)` directly without adding the CFA baseline). This file is pending removal.

All code must import from the root-level `model.py`:

```python
# Correct
from model import XTransUNet, count_parameters

# Wrong — legacy, no base_width support
from src.model import XTransUNet
```

If you encounter imports from `src.model` or `src/model.py`, migrate them to the root `model.py`.

### Do Not Hardcode Channel Widths

When reconstructing a model for inference, always read `base_width` from the checkpoint rather than assuming the default of 64. Checkpoints trained with `base_width=32` or `base_width=16` will produce incorrect results or crash if loaded into a model instantiated with the wrong width.

```python
# Correct
model = XTransUNet(base_width=ckpt.get("base_width", 64))

# Wrong — assumes default width
model = XTransUNet()
```

### Do Not Modify the Residual Skip Structure

The `baseline + self.out_conv(d1)` pattern in the forward method is load-bearing for exposure agnosticism and training stability. Do not remove the residual skip, change the baseline construction (e.g., reverting to broadcast instead of per-channel masking, or averaging channels), or move the addition to an intermediate layer. Any such change would invalidate all existing checkpoints and require full retraining.
