---
title: Composite Demosaicing Loss Functions
tags: [training, loss, optimization]
scope: losses.py
generated: 2026-03-22
commit: 347c8dd
---

## Context

A single loss function cannot capture every dimension of demosaicing quality. L1 loss alone optimizes for PSNR but produces blurry textures. SSIM alone can allow color shifts. Neither explicitly penalizes the false-color zipper artifacts that are the hallmark failure mode of demosaicing, nor do they account for the structural asymmetry of the X-Trans color filter array where green photosites outnumber red and blue by roughly 2.5:1.

`DemosaicLoss` in `losses.py` is a weighted composite that combines six loss components, each targeting a different quality dimension. The composite feeds gradients to the `XTransUNet` model described in [unet-model-architecture](unet-model-architecture.md). All components operate on 3-channel RGB tensors in `(B, C, H, W)` layout, with pixel values in `[0, data_range]`.

## Pattern / Approach

### Loss Components

#### L1 / Huber Pixel Loss

The primary pixel-accuracy term. Computes mean absolute error between prediction and target across all spatial positions and channels. This is the dominant driver of PSNR.

When `use_huber=True`, the loss switches to `F.huber_loss` with a configurable `huber_delta` (default 1.0). Huber loss is less sensitive to outlier pixels (e.g., specular highlights, hot pixels), reducing gradient explosion on extreme residuals while behaving like L1 in the normal range.

**Per-channel normalization** (`per_channel_norm=True`): Instead of computing L1 over the entire `(B, 3, H, W)` tensor, the loss is computed separately for R, G, and B channels and then averaged with equal weight:

```
pixel_loss = (loss_R + loss_G + loss_B) / 3
```

Without per-channel normalization, `F.l1_loss` averages over all elements. Because X-Trans CFA has ~20 green photosites per 36-site tile versus ~8 red and ~8 blue, training data where the network learns from CFA-masked inputs will naturally weight green errors more heavily in the gradient. Per-channel normalization ensures each channel contributes equally to the loss regardless of CFA sample distribution. The per-channel components (`l1_r`, `l1_g`, `l1_b`) are logged individually for monitoring convergence balance.

#### MS-SSIM (Multi-Scale Structural Similarity)

Implemented in the `MSSSIM` class. Computes structural similarity at 5 scales by progressively downsampling with 2x average pooling, following Wang et al. 2003. At each scale, the luminance and contrast-structure components are extracted using Gaussian-windowed statistics (window size 11, sigma 1.5). The default per-scale weights are `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`.

At intermediate scales, only the contrast-structure component is used. At the final (coarsest) scale, the luminance component is also included. This matches the original MS-SSIM formulation: fine scales capture local texture and edges, coarse scales capture global luminance and contrast.

MS-SSIM returns a similarity value in `[0, 1]` (higher is better). `DemosaicLoss` converts it to a loss via `1 - msssim_val` so that the optimizer minimizes it. The raw similarity value is logged in the `msssim` component for monitoring.

The stability constant calculation depends on `data_range`:

```
C1 = (0.01 * data_range) ** 2
C2 = (0.03 * data_range) ** 2
```

When training on linear-light data (pre-gamma, potentially with white-balance multipliers), the effective pixel range can exceed 1.0. The `data_range` parameter must match the actual maximum pixel value for the SSIM constants to be numerically correct. In `train.py`, `data_range` is auto-computed from dataset metadata when `--apply-wb` is set, or defaults to 1.0.

If the input tensor becomes too small at a given scale (either dimension < `window_size`), that scale and all subsequent scales are skipped. This can happen with small patch sizes.

#### Sobel Gradient Loss

Implemented in `SobelGradientLoss`. Applies 3x3 Sobel filters (horizontal and vertical) to each channel independently, then compares prediction and target gradients with L1 loss on each component: `(|gx_pred - gx_target| + |gy_pred - gy_target|).mean()`.

The Sobel kernels are registered as buffers (not parameters) and move to the correct device automatically. The convolution reshapes `(B, C, H, W)` to `(B*C, 1, H, W)` to apply the single-channel Sobel kernel across all channels efficiently.

This component directly penalizes edge smearing and ringing. A network optimized purely on L1 tends to produce smooth transitions at edges because the L1-optimal prediction at an uncertain edge is the mean of possible edge positions. The gradient loss forces the network to commit to sharp edge placement even when this slightly hurts PSNR.

#### Chroma Artifact Loss

Implemented in `ChromaLoss`. Targets false-color artifacts (zipper patterns, color fringing) that are the most visually objectionable demosaicing failure mode.

The approach:

1. Convert RGB to chrominance (Cb, Cr) using simplified YCbCr coefficients. The luminance channel is discarded since legitimate image detail lives in luminance.
2. Apply a Gaussian low-pass filter (default kernel size 5, sigma = kernel_size / 4) to the chrominance channels.
3. Compute the high-frequency residual: `highfreq = original - lowpass`.
4. Compare the high-frequency chrominance of prediction and target with L1 loss.

The key insight: in natural images, chrominance is spatially smooth. High-frequency chrominance content is almost always a demosaicing artifact, not real scene content. By specifically penalizing high-frequency chroma differences, this loss catches zipper and false-color artifacts that L1 loss treats as just another pixel error.

#### Color Bias Loss

Implemented in `ColorBiasLoss`. Computes the spatial mean of each channel (giving a `(B, 3)` tensor of per-image channel means) and penalizes the L1 difference between prediction and target means.

This is a DC-component penalty: it prevents the network from developing a systematic color shift across the entire image. A network that is slightly but consistently too warm or too cool across all images will be penalized. This is orthogonal to the per-pixel L1 loss because a small DC bias contributes negligibly to per-pixel error when spread across many pixels, but is visually noticeable as a color cast.

#### Zipper Loss

Implemented in `ZipperLoss`. Penalizes spurious high-frequency oscillations (zipper artifacts) using the discrete Laplacian (2nd-order derivative). The 3x3 Laplacian kernel `[[0,1,0],[1,-4,1],[0,1,0]]` is convolved with each channel of both prediction and target, and the L1 difference between the two Laplacian responses is the loss.

The Laplacian detects rapid sign alternation -- the defining characteristic of zipper artifacts, where adjacent pixels oscillate between high and low values. The first-order Sobel gradient loss already penalizes edge errors, but zipper is specifically a second-order phenomenon that Sobel largely misses. By operating on the Laplacian response rather than raw pixel values, this loss has high sensitivity to the alternating patterns that zipper artifacts produce.

Like `SobelGradientLoss`, the convolution reshapes `(B, C, H, W)` to `(B*C, 1, H, W)` and applies the single-channel Laplacian kernel across all channels. The kernel is registered as a buffer.

### DemosaicLoss Composition

`DemosaicLoss` is the unified wrapper. Its constructor takes a weight for each component:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `l1_weight` | 1.0 | Weight for L1/Huber pixel loss |
| `msssim_weight` | 0.0 | Weight for `1 - MS-SSIM` |
| `gradient_weight` | 0.1 | Weight for Sobel gradient loss |
| `chroma_weight` | 0.05 | Weight for chroma artifact loss |
| `color_bias_weight` | 0.0 | Weight for color bias loss |
| `zipper_weight` | 0.0 | Weight for Laplacian zipper loss |
| `per_channel_norm` | False | Per-channel L1 normalization |
| `use_huber` | False | Use Huber instead of L1 |
| `huber_delta` | 1.0 | Huber loss delta parameter |
| `data_range` | 1.0 | Pixel value range for SSIM constants |
| `recon_only` | False | Compute L1/Huber only on reconstructed (non-sampled) pixels |
| `known_pixel_weight` | 0.1 | Weight for known-pixel penalty when `recon_only` is True |

Components with weight 0 are not instantiated (the corresponding submodule is `None`), so they add zero overhead. The forward pass signature is `forward(pred, target, clip_levels=None, channel_masks=None)`, where `channel_masks` is a `(B, 3, H, W)` binary tensor needed for `recon_only` mode. It returns a tuple of `(total_loss, components_dict)` where the dict contains the raw (unweighted) value of each active component plus the weighted total. This enables per-component monitoring in training logs.

The total loss is computed as:

```
total = l1_weight * pixel_loss
      + msssim_weight * (1 - msssim)
      + gradient_weight * gradient_loss
      + chroma_weight * chroma_loss
      + zipper_weight * zipper_loss
      + color_bias_weight * color_bias_loss
```

### Reconstruction-Only Mode

When `recon_only=True` and a `channel_masks` tensor is provided to the forward pass, the L1/Huber pixel loss is split into two parts:

- **Reconstructed pixels** (where `channel_masks == 0`): the primary loss, computed on pixels that the network must interpolate from CFA neighbors.
- **Known pixels** (where `channel_masks == 1`): a small penalty weighted by `known_pixel_weight` (default 0.1), ensuring the network does not drift at positions where the CFA directly sampled the channel.

The `channel_masks` tensor has shape `(B, 3, H, W)` with binary values -- 1 where the CFA samples that channel at that spatial position, 0 elsewhere. The forward method uses `_masked_loss` to compute mean loss over each mask region separately, then combines them:

```
pixel_loss = recon_loss + known_pixel_weight * known_loss
```

Both `{l1|huber}_recon` and `{l1|huber}_known` are logged as separate components for monitoring. The structural losses (MS-SSIM, gradient, chroma, zipper, color bias) still operate on the full image regardless of `recon_only`.

If `recon_only=True` but `channel_masks` is not passed, the mode has no effect and the loss falls through to the standard (or per-channel) pixel loss path. Note that `recon_only` (when active) and `per_channel_norm` are mutually exclusive code paths -- when the recon mask is active, per-channel normalization is not applied to the pixel loss.

### Data Range Handling

SSIM and MS-SSIM stability constants are functions of the data range. The default `data_range=1.0` assumes pixel values normalized to `[0, 1]`. When training with white-balance multipliers applied (`--apply-wb` in `train.py`), pixel values can exceed 1.0 because WB gains amplify certain channels. In this case, `train.py` computes the effective data range from dataset metadata and passes it to `DemosaicLoss`.

Other loss components (L1, gradient, chroma, color bias) are scale-agnostic because they use L1 distance, which scales linearly with data range. Their weights may need adjustment when changing data range, but they do not have internal constants that depend on it.

### Preset Constructors

`DemosaicLoss` provides two class methods as convenience constructors:

- **`DemosaicLoss.base(data_range=1.0)`** -- L1-focused preset: `l1=1.0, gradient=0.1, chroma=0.05, zipper=0.05, msssim=0.0`. Prioritizes PSNR.
- **`DemosaicLoss.finetune(msssim_weight=0.3, gradient_weight=0.2, data_range=1.0)`** -- Perceptual preset: `l1=0.5, msssim=0.3, gradient=0.2, chroma=0.02, zipper=0.1`. Adds MS-SSIM and increases gradient and zipper weights at the expense of L1.

**These are convenience starting points, not canonical training configurations.** In `train.py`, the `--mode` flag selects which preset to start from, but all individual weights can be overridden via CLI arguments (`--l1-weight`, `--msssim-weight`, `--gradient-weight`, `--chroma-weight`, `--color-bias-weight`, `--zipper-weight`, `--per-channel-norm`, `--huber`, `--huber-delta`, `--recon-only`, `--known-pixel-weight`). Actual training checkpoints were trained with explicit CLI arguments specifying individual weights based on ongoing experimentation. Do not treat the preset values as the weights used for any particular checkpoint.

`train.py` applies CLI overrides by mutating the criterion object after construction. When a component that was initially disabled (weight 0, submodule `None`) is enabled via CLI override, `train.py` lazily instantiates the submodule:

```python
if criterion.msssim is None and args.msssim_weight > 0:
    criterion.msssim = MSSSIM(data_range=data_range).to(device)
```

The same lazy-instantiation pattern applies to `ColorBiasLoss` and `ZipperLoss`.

## Rationale

### Per-Channel Normalization for X-Trans Green Imbalance

The X-Trans 6x6 CFA tile contains 20 green, 8 red, and 8 blue photosites. When computing L1 loss over the full `(B, 3, H, W)` output tensor, each pixel contributes equally. But during training, the network's ability to reconstruct each channel is influenced by how much CFA information is available for that channel. Green, with 2.5x more samples, is inherently easier to reconstruct. Without per-channel normalization, the gradient signal is dominated by whichever channel has the largest absolute errors, and the optimizer may underinvest in green quality (since green errors are small) or overinvest in reducing red/blue errors at the expense of overall balance. Per-channel normalization ensures the optimizer treats each channel as equally important regardless of the CFA sampling ratio.

### Gradient Loss for Edge Preservation

Demosaicing is an interpolation problem. At edges, the network must decide whether adjacent pixels belong to the same surface or different surfaces. L1 loss penalizes the average error, so the L1-optimal prediction at an ambiguous edge is a blended value. The gradient loss penalizes the difference in edge magnitude between prediction and target, directly rewarding the network for producing edges with the correct sharpness and position. This is especially important for X-Trans because the irregular CFA pattern creates different interpolation contexts at horizontal, vertical, and diagonal edges.

### Chroma Loss for False-Color Suppression

False-color artifacts are the most visually offensive demosaicing failure. They appear as colored fringes along high-contrast edges where the CFA provides insufficient color information. A small amount of false color can be nearly invisible in PSNR (a few pixels off by a moderate amount) but highly visible to the human eye because the visual system is sensitive to unexpected chrominance at edges. The chroma loss specifically targets this failure mode by operating in chrominance space and focusing on high-frequency content, giving it far more leverage per unit of loss weight than L1 alone.

### MS-SSIM for Perceptual Quality

Single-scale SSIM captures local structural similarity but misses quality differences at different spatial frequencies. MS-SSIM evaluates structure at 5 scales, which better correlates with human perception of texture detail. It is particularly useful in fine-tuning stages where PSNR is already high and the remaining quality gains are in texture fidelity and micro-contrast -- qualities that L1 loss is nearly blind to.

### Zipper Loss for Second-Order Artifact Suppression

Zipper artifacts are alternating bright/dark pixel patterns along edges where the demosaicing algorithm oscillates between over- and under-interpolation. Sobel gradient loss detects first-order edge errors (wrong edge strength or position) but is largely insensitive to the second-order oscillation that defines zipper. The Laplacian-based ZipperLoss specifically targets this: a smooth edge has low Laplacian response, while a zippered edge produces high Laplacian response due to rapid curvature changes. Penalizing the difference in Laplacian response between prediction and target forces the network to produce smooth interpolations rather than oscillating ones.

### Reconstruction-Only Mode for Targeted Optimization

When the network receives CFA-masked input, the known pixel positions (where the CFA directly sampled a channel) are essentially identity-mapping targets -- the network should reproduce them exactly. Spending gradient budget on these easy pixels is wasteful when the challenge is in the unknown (interpolated) positions. `recon_only` mode focuses the L1/Huber loss on the pixels that the network must actually reconstruct, while the small `known_pixel_weight` penalty prevents the network from ignoring known positions entirely (which could cause drift in skip-connection pathways).

## Key Files

| File | Role |
|------|------|
| `losses.py` | `DemosaicLoss`, `MSSSIM`, `SSIM`, `SobelGradientLoss`, `ChromaLoss`, `ZipperLoss`, `ColorBiasLoss` |
| `train.py` | Instantiates `DemosaicLoss` via preset, applies CLI weight overrides, logs per-component losses |
| `ui.py` | Reads loss component history for visualization; converts MS-SSIM similarity to loss (`1 - v`) for plotting |

## Antipatterns

**Do not import from `src/losses.py`.** The file `src/losses.py` is a legacy version containing `CombinedLoss`, `ChromaticArtifactLoss`, and `SSIMGradientLoss`. It predates the unified `DemosaicLoss` API, lacks MS-SSIM support, has no `data_range` parameter (hardcodes SSIM constants for `[0, 1]` range), and does not support per-channel normalization or Huber loss. All new code should import from the top-level `losses.py`. The alias `CombinedLoss = DemosaicLoss` exists in `losses.py` for backward compatibility but should not be used in new code.

**Do not assume preset weights match checkpoint weights.** The `base()` and `finetune()` class methods encode reasonable starting points, but every released training checkpoint was trained with explicit per-component weights specified via CLI arguments. Reading the preset defaults to infer what weights a checkpoint was trained with will give incorrect values. Check the checkpoint's saved config or the training command that produced it.

**Do not change `data_range` without updating loss weights.** While L1, gradient, chroma, and color bias losses scale linearly with data range, the absolute loss values change. Loss weights that produce good results at `data_range=1.0` will produce different gradient magnitudes at `data_range=2.5`. When switching between normalized and white-balanced training data, re-tune loss weights or at minimum verify that the component loss magnitudes remain in a reasonable ratio.
