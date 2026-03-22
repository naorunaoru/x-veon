---
title: Synthetic Torture Test Patterns
tags: [dataset, training, torture-tests, synthetic]
scope: torture_v2.py
generated: 2026-03-22
commit: 347c8dd
---

## Context

Demosaicing networks trained exclusively on natural photographs develop blind spots. Natural images rarely contain perfectly regular high-frequency structure at or near the Nyquist limit of the CFA (Color Filter Array). When the network encounters such structure in real sensor data -- fine fabric weave, roof tiles at a distance, striped shirts, mesh screens -- it produces moire, false color, and zipper artifacts because it never learned to reconstruct those spatial frequencies faithfully.

Synthetic torture test patterns address this gap by injecting worst-case geometric and chromatic stimuli into the training set. Each pattern isolates a specific failure mode: Nyquist-frequency grids test aliasing resistance, chromatic edges test cross-channel interpolation, Siemens stars test orientation-invariant resolution, and fractal textures test handling of detail at multiple spatial scales simultaneously. By mixing these patterns into training data at a low fraction (typically 5%), the network learns to handle edge cases without overfitting to synthetic structure.

This is the v2 system. The original torture test generator (`TortureTestLinearDataset` in `src/datasets/dataset_v4.py`) provided only 5 pattern types with no supersampling and naive random colors. The v2 system in `torture_v2.py` expands to 13 pattern types, adds 4x supersampling for alias-free ground truth, introduces a multi-strategy color picker to exercise cross-channel reconstruction, and includes fractal generators for organic high-frequency content.

Prerequisite: the [dataset-build](dataset-build.md) document describes the real-image dataset pipeline that these synthetic patterns are mixed into.

## Pattern/Approach

### TortureDatasetV2 Class

`TortureDatasetV2` is a PyTorch `Dataset` defined in `torture_v2.py`. It generates patterns procedurally at access time -- no files on disk are required, though a `generate_dataset()` utility can pre-render patterns to `.npz` files.

Constructor parameters:

| Parameter | Default | Purpose |
|---|---|---|
| `size` | 288 | Output patch dimensions (square) |
| `num_patterns` | 500 | Total patterns in the dataset |
| `add_noise` | `True` | Apply simulated sensor noise |
| `noise_prob` | 0.5 | Probability of noise per sample |
| `cfa_type` | `"xtrans"` | CFA pattern key from `CFA_REGISTRY` |

Each call to `__getitem__(idx)` produces a `(input_tensor, rgb)` pair where `input_tensor` is a 4-channel tensor (CFA mosaic + 3 channel masks) and `rgb` is the 3-channel linear ground truth. The pattern type is selected deterministically as `idx % 13`, so every 13th sample cycles through all pattern types.

### 4x Supersampling Strategy

All patterns are rendered at 4x the output resolution (`size * 4` in each dimension), then downsampled to the target size using area interpolation (`F.interpolate` with `mode="area"`). This is critical for two reasons:

1. **Alias-free ground truth.** Many patterns contain hard binary edges (stripes, checkerboards, grids). Rendering these directly at output resolution would produce ground-truth images with aliased staircase edges. The network would then learn to reproduce aliasing rather than smooth reconstruction.

2. **Sub-pixel feature representation.** Patterns whose spatial frequency approaches or exceeds 1 cycle per 2 pixels at output resolution (the Nyquist limit) need supersampled rendering to correctly represent the energy distribution across pixels. Without supersampling, a 1-pixel-wide stripe that falls between pixel centers disappears entirely, producing misleading training signal.

The 4x factor balances quality against memory cost. At `size=288`, the hi-res buffer is 1152x1152 per channel. The `SUPERSAMPLE = 4` class constant controls this factor.

Pattern parameters that depend on spatial frequency (stripe period, grid cell size, line width) are scaled by the supersampling factor so their effective frequency at the output resolution matches the intended target. For example, a `line_width` of `rng.randint(1, 12) * ss` in hi-res coordinates produces lines between 1 and 12 pixels wide at the output resolution after area downsampling.

### The 13 Pattern Types

Pattern type is determined by `idx % 13`. All geometric patterns support arbitrary rotation via a `rotate_coords()` helper that rotates coordinate grids around the image center.

| Index | Name | Description |
|---|---|---|
| 0 | **Diagonal stripes** | Sinusoidal stripes at random frequency and angle. Tests orientation-dependent aliasing. Frequency range: 0.02-0.4 cycles/pixel (scaled by 1/ss). |
| 1 | **Parallel lines** | Binary line pairs at integer-pixel widths (1-12px at output res). Tests reconstruction of hard edges at sub-CFA-period widths. |
| 2 | **Checkerboard** | Rotated checkerboard with cell sizes 2-24px. Exercises the worst-case 2D Nyquist pattern where every adjacent pixel differs. |
| 3 | **Concentric circles** | Sinusoidal rings radiating from center. Tests radially symmetric frequency content that sweeps from low to high frequency. |
| 4 | **Color gradient** | Linear ramp between two colors along a rotated axis. Tests smooth inter-channel interpolation with no hard edges. |
| 5 | **Siemens star** | Radial spoke pattern (8-36 spokes). Frequency increases toward the center, providing a continuous resolution test across all orientations in a single image. |
| 6 | **Slanted edge** | Single rotated binary edge. The standard test target for measuring MTF (Modulation Transfer Function). Tests edge reconstruction accuracy at arbitrary angles. |
| 7 | **Nyquist grid** | Checkerboard or line grid at 1-2px cell size -- at or below the CFA period. This is the most aggressive aliasing stimulus. Either 1D or 2D form is selected randomly. |
| 8 | **Chromatic edges** | Rotated binary edge using forced complementary color pairs. Tests cross-channel bleed at sharp color transitions, which causes false color and zipper artifacts. |
| 9 | **Radial color wheel** | Hue mapped to angle, with radial intensity falloff. Tests smooth hue reconstruction when the dominant channel changes continuously. |
| 10 | **Brick pattern** | Offset rectangular grid (brickwork layout) at variable cell sizes. Tests reconstruction of periodic structure with half-period offset between rows. |
| 11 | **Julia set fractal** | Julia set iteration with 8 base parameter sets, each jittered randomly. 50% chance of deep zoom (8-40x) into boundary regions found by a sampling search (`find_julia_boundary_point`). Produces organic, non-periodic high-frequency structure at multiple scales. |
| 12 | **Perlin noise fractal** | Fractal Brownian motion (3-6 octaves, persistence 0.4-0.7). Produces smooth multi-scale organic texture that resembles natural surfaces. |

Patterns 0-10 are geometric and binary-masked: they paint two colors through a spatial mask. Patterns 11-12 are fractal and produce continuous-valued imagery mapped to color through `_fractal_to_rgb()`.

### Smart Color Picker

The `ColorPicker` class selects color pairs for geometric patterns using four strategies, chosen probabilistically in "mixed" mode:

| Strategy | Probability | Purpose |
|---|---|---|
| **HSV random** | 40% | Two random HSV colors with a minimum Euclidean contrast of 0.15 in linear RGB. Falls back to complementary hues after 20 failed attempts. Ensures broad coverage of the color gamut. |
| **Complementary** | 20% | Hand-curated pairs where one channel is strong and the other two are weak (e.g., red vs. cyan). Forces the network to reconstruct opposite sides of the color wheel at a hard boundary. |
| **Cross-channel** | 20% | Pairs where adjacent CFA channels carry conflicting information (red vs. green, green vs. blue). Targets the inter-channel crosstalk that causes false color. |
| **Near-neutral** | 20% | Subtly different near-gray pairs (e.g., `[0.20, 0.18, 0.18]` vs. `[0.18, 0.20, 0.18]`). Forces the network to distinguish minute chromatic differences that would be masked by quantization or noise. |

All values are in linear RGB space in the 0.02-0.6 range, consistent with the sensor value ranges in the real training data (see [dataset-build](dataset-build.md)). Each curated pair is jittered by +/-0.03 per channel to prevent memorization of exact color values.

HSV-to-linear conversion applies a gamma of 2.2 (`c ** 2.2`) to match the linear domain of the training pipeline.

### Fractal Generators

Two fractal generators produce non-periodic, multi-scale structure:

**Julia set** (`julia_set()`): Iterates `z = z^2 + c` on a complex grid. Returns normalized escape-time values. The `find_julia_boundary_point()` function samples 500 random points and scores them by how close their escape iteration is to the midpoint of `max_iter`, identifying boundary regions where detail is richest. Deep zoom factors (8-40x) into these boundary regions produce extremely fine structure that approaches the Nyquist limit.

Eight base Julia parameter sets are defined in `JULIA_PARAMS`, each producing a distinct fractal shape. Parameters are jittered by +/-0.05 to prevent exact repetition.

**Perlin noise** (`fractal_noise()`): Multi-octave Perlin noise (fractal Brownian motion). Each octave uses a separate seed derived from the base seed offset by `i * 1000`. The result is normalized to [0, 1]. Base frequency ranges from 0.01 to 0.04, with 3-6 octaves and persistence 0.4-0.7.

Both fractals are converted to RGB via `_fractal_to_rgb()`, which offers three coloring modes selected randomly:

- **hue**: Maps fractal value to a cyclic hue rotation, producing smooth rainbow gradients.
- **gradient**: Linear interpolation between two random HSV colors, producing a two-tone colormap.
- **bands**: Quantizes the fractal into discrete bands (4-12 levels) alternating between two complementary colors, producing sharp edges at fractal contours.

### Per-Pattern Deterministic RNG

Each pattern uses a dedicated `random.Random(idx * 7919)` instance, where 7919 is a prime multiplier. This guarantees:

- **Reproducibility**: the same index always produces the same pattern, regardless of access order or parallelism.
- **No cross-pattern interference**: each pattern's random choices (angle, frequency, colors, fractal parameters) are independent.
- **Training stability**: identical patterns across epochs, so the network sees consistent ground truth for each index.

The multiplier 7919 spaces seeds far apart in the RNG state space, avoiding correlations between adjacent indices.

### Optional Sensor Noise

When `add_noise=True` (default), each pattern has a `noise_prob` (default 0.5) chance of receiving simulated sensor noise. The noise model is signal-dependent:

```
noise_std = sqrt(shot_sigma * rgb + read_sigma^2)
```

- `read_sigma`: uniform in [0.005, 0.02] -- models read noise (signal-independent).
- `shot_sigma`: uniform in [0.01, 0.05] -- models photon shot noise (signal-dependent, Poisson-like).

This is a signal-dependent Poisson-Gaussian model, distinct from the real data augmentation pipeline which uses simple additive Gaussian noise applied after mosaicing to the CFA input only. In torture patterns, noise is applied before CFA mosaicing to the full RGB image, so both the mosaiced input and the ground truth target contain the same noise.

### Integration with Training

The torture patterns integrate into training through a three-layer delegation chain:

1. **`TortureDatasetV2`** (`torture_v2.py`) -- generates patterns, applies CFA mosaic, returns `(input_tensor, rgb)`.
2. **`TortureDataset`** (`dataset.py`) -- thin wrapper that imports and delegates to `TortureDatasetV2`. Exists so `dataset.py` consumers do not need a direct import of `torture_v2`.
3. **`create_mixed_dataset()`** (`dataset.py`) -- combines a `LinearDataset` of real images with a `TortureDataset`, wrapped in `ConcatDataset`. The torture fraction is controlled by:
   - `torture_fraction` (default 0.05): desired proportion of synthetic patterns in the combined dataset.
   - `torture_patterns` (default 500): number of unique patterns. The `RepeatedDataset` inner class wraps the torture dataset with modular indexing to reach the target size, capped at 10x the unique pattern count.

The `train.py` CLI exposes `--torture-fraction` and `--torture-patterns` arguments. When `--torture-fraction` is greater than 0, the training loop calls `create_mixed_dataset()` instead of creating a bare `LinearDataset`. Typical invocation:

```
python train.py --data-dir /path/to/npy --torture-fraction 0.05 --torture-patterns 500
```

This produces a combined dataset where roughly 5% of batches contain synthetic patterns and 95% contain real image patches.

## Rationale

### Why These Specific Patterns

Each pattern targets a known failure mode of demosaicing algorithms:

- **Stripes, lines, grids, checkerboard**: periodic structure that aliases against the CFA sampling grid. The X-Trans 6x6 pattern and Bayer 2x2 pattern each have characteristic aliasing frequencies; these patterns sweep through that frequency range.
- **Siemens star**: provides a continuous frequency sweep across all orientations in one image. If the network handles a Siemens star well, it handles edges at any angle.
- **Slanted edge and chromatic edge**: isolate the single-edge case that determines the MTF. Chromatic edges specifically target false-color artifacts.
- **Color gradient and radial wheel**: test smooth interpolation. A network that handles hard edges but produces banding on gradients has learned to threshold rather than interpolate.
- **Fractals**: provide non-periodic, multi-scale structure that is closer to natural texture than geometric patterns but still contains high-frequency detail at predictable locations. Julia sets in particular produce filament structure that is geometrically similar to hair, fabric threads, and vegetation edges.
- **Brick pattern**: tests reconstruction of offset periodic structure, common in real-world architecture.

### Why 4x Supersampling

The CFA period is 6 pixels for X-Trans and 2 pixels for Bayer. Patterns at the Nyquist limit produce features as small as 2 pixels (Bayer) or 6 pixels (X-Trans) in the output. Rendering these features directly at output resolution produces hard aliased edges in the ground truth, which means the network learns to reconstruct aliased output -- the opposite of the intended goal.

4x supersampling means the rendering grid is 4x finer than the output grid. A 1-pixel output feature spans 4 rendering pixels, giving the area downsampler enough information to produce a correct partial-coverage value. This is particularly important for Nyquist grid (pattern 7), where a 1px cell at output resolution must produce a spatially correct checkerboard after downsampling.

2x supersampling was tested and found insufficient: binary features at odd pixel positions produced asymmetric coverage. 8x was tested and found to produce negligible quality improvement at 4x the memory cost. 4x is the practical optimum.

### Why Smart Color Pairing

Naive random RGB colors cluster around mid-gray with similar channel values. This is because sampling three independent uniform random values tends toward balanced mixtures. The result: most training patterns have low cross-channel contrast, so the network rarely practices reconstructing extreme color transitions.

The `ColorPicker` strategies correct this by:

- **Complementary pairs**: guarantee large cross-channel difference (one channel high, two low).
- **Cross-channel pairs**: guarantee that adjacent CFA channels carry conflicting information.
- **Near-neutral pairs**: guarantee the network must resolve tiny chromatic differences that would be easy to hallucinate away.
- **HSV random**: covers the remaining gamut with a minimum contrast threshold.

## Key Files

| File | Role |
|---|---|
| `torture_v2.py` | All pattern generation, `TortureDatasetV2` class, `ColorPicker`, fractal generators, `generate_dataset()` and `generate_samples()` utilities. |
| `dataset.py` | `TortureDataset` wrapper (lines 207-221), `create_mixed_dataset()` function (lines 224-285) that concatenates real and synthetic data. |
| `train.py` | CLI arguments `--torture-fraction` and `--torture-patterns`; conditional `create_mixed_dataset()` call. |
| `cfa.py` | `CFA_REGISTRY`, `make_cfa_mask()`, `make_channel_masks()` used by `TortureDatasetV2` to produce CFA-mosaiced inputs. |

## Antipattern

### Do Not Use TortureTestLinearDataset (v1)

The legacy `TortureTestLinearDataset` in `src/datasets/dataset_v4.py` has only 5 pattern types, no supersampling, no smart color selection, and no fractal patterns. It generates patterns directly at output resolution, producing aliased ground truth that teaches the network to reproduce staircase edges. Always use `TortureDatasetV2` from `torture_v2.py` via the `TortureDataset` wrapper in `dataset.py`.

### Do Not Set Torture Fraction Above 10%

At fractions above 10%, the synthetic patterns begin to dominate the loss landscape. The network overfits to binary geometric structure and loses the ability to reconstruct natural texture gradients, subtle color variation, and sensor-specific noise characteristics present in real data. The default of 5% was chosen empirically to improve worst-case metrics (Nyquist star, chromatic edge) without degrading average PSNR on natural images.

### Do Not Disable Supersampling

Setting `SUPERSAMPLE = 1` or removing the `_downsample()` step produces aliased ground truth. The network will learn to reproduce aliasing artifacts because the supervision signal contains them. The 4x factor is load-bearing for pattern correctness.

### Do Not Bypass the ColorPicker for Random RGB

Replacing `ColorPicker` with independent `torch.rand(3)` values for each color results in a narrow distribution centered on mid-gray with low cross-channel contrast. The network will appear to train normally but will produce false color on saturated edges in real images because it never saw high cross-channel contrast during training.
