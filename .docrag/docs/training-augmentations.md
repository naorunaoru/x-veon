---
title: Training Data Augmentation Strategy
tags: [dataset, training, augmentation, degradation, cfa]
scope: dataset augmentation pipeline in LinearDataset._process_patch / __getitem__
generated: 2026-03-22
commit: 347c8dd
---

## Context

Demosaicing neural networks reconstruct full RGB images from spatially subsampled CFA
(Color Filter Array) sensor data. The model must generalize across a wide range of
real-world conditions: varying white balance, exposure levels, sensor noise profiles,
and optical blur characteristics. Without augmentation, the network overfits to the
narrow distribution of the training corpus and fails on out-of-distribution inputs at
inference time.

All augmentations in x-veon are **CFA-aware**. Every spatial transform must respect
the periodicity of the CFA pattern (6x6 for X-Trans, 2x2 for Bayer) to avoid
introducing impossible pattern alignments that would never appear in real sensor data.
This constraint eliminates most of the geometric augmentations common in general
computer vision (arbitrary-angle rotations, non-aligned crops, shears). Discrete
90-degree rotations and axis flips are safe when applied before mosaicing.

The augmentation pipeline lives in `LinearDataset._process_patch` (called from
`__getitem__`) in `dataset.py`. It operates on pre-computed linear `.npy` image files
produced by the dataset build pipeline (see [dataset-build](dataset-build.md) for how
these are created). CFA mask generation and period calculations come from the pattern
system documented in [cfa-pattern-system](cfa-pattern-system.md).

The dataset returns a **3-tuple** `(input_tensor, ref, clip_ch)`:
- `input_tensor`: `(5, H, W)` — mosaiced CFA + 3 channel masks + clip ratio channel
- `ref`: `(3, H, W)` — ground truth RGB (captured after flips/rotations, before OLPF)
- `clip_ch`: `(3,)` — per-channel clip levels (`wb * clip_scale`) for loss masking

The pipeline's transforms fall into two categories. **Augmentations** (steps 1–5)
modify both the input and the ground-truth reference equally, increasing the effective
diversity of scenes and conditions. **Degradations** (steps 6, 8, 10) corrupt only
the network input while the reference stays clean, teaching the network to invert a
physical corruption (blur, clipping, noise). The dividing line is the point where `ref`
is captured — after geometric transforms (step 5) but before OLPF blur (step 6).

## Pattern / Approach

### Augmentation Pipeline Order

The augmentations execute in a fixed order within `_process_patch` (plus crop/downscale
in `__getitem__`). The ordering is deliberate and physically motivated — it mirrors the
signal chain of a real camera sensor:

1. **CFA-aligned random crop** — extract a patch from the full image (`__getitem__`)
2. **Area-average downscale** — optional 2x downscale (`__getitem__`, before `_process_patch`)
3. **White balance application + WB perturbation** — apply and jitter WB gains
4. **Bright-spot augmentation** — add synthetic point light sources (before flips)
5. **Spatial flips + 90-degree rotations** — geometric augmentation (CFA-safe)
   **↓ `ref` captured here — augmentations above, degradations below ↓**
6. **OLPF blur** — simulate optical low-pass filter (before mosaicing)
7. **Mosaicing** — apply CFA pattern to produce single-channel sensor data
8. **Sensor saturation clipping** — clamp CFA values to per-channel clip levels (bright-spot patches only)
9. **Clip ratio computation** — encode proximity to sensor saturation as a 0-1 ramp (always computed)
10. **Poisson-Gaussian noise injection** — simulate sensor read + shot noise (after mosaicing)
11. **5-channel input assembly** — concatenate CFA image, channel masks, and clip ratio

The ground-truth reference (`ref`) is captured **before** OLPF blur (step 6) and
**after** flips/rotations (step 5). Steps 1–5 are **augmentations** (applied equally
to input and reference), while steps 6, 8, and 10 are **degradations** (applied only
to the input). The network learns to reconstruct clean, sharp RGB from a degraded
(blurred, clipped, noisy) mosaiced input.

### 1. CFA-Aligned Random Crops

```
top = (rng.randint(0, max(0, max_y)) // self.period) * self.period
left = (rng.randint(0, max(0, max_x)) // self.period) * self.period
```

Random crop coordinates are snapped to multiples of the CFA period (`self.period` —
6 for X-Trans, 2 for Bayer). This guarantees that every extracted patch starts at the
top-left corner of a complete CFA tile. Without this alignment, the CFA mask
(pre-computed once at dataset construction time) would be out of phase with the actual
color filter positions in the cropped patch.

The patch size itself is validated at construction time to be divisible by
`patch_alignment()`, which returns `lcm(cfa_period, 16)` — the least common multiple
of the CFA period and the UNet's spatial downsampling factor. For X-Trans this is
`lcm(6, 16) = 48`, so valid patch sizes are 48, 96, 144, 192, etc.

Each image yields `patches_per_image` random crops per epoch (default 16), giving the
effective dataset size of `len(files) * patches_per_image`.

### 2. Area-Average Downscale Augmentation

```
do_downscale = (self.augment and self.downscale_prob > 0
                and rng.random() < self.downscale_prob)
crop_size = self.patch_size * 2 if do_downscale else self.patch_size
...
rgb = rgb.view(3, crop_size // 2, 2, crop_size // 2, 2).mean(dim=(2, 4))
```

**Parameter:** `--downscale-prob` (default 0.0, typical value 0.0–0.5)

Crops a patch at 2x the target resolution and area-averages it down to the target
patch size. This teaches the network to handle lower-resolution content (distant
subjects, high-megapixel sensors downsampled for display) where the effective CFA
pattern is coarser relative to scene detail.

The 2x crop is taken first in `__getitem__`, with the same CFA-aligned random crop
logic. If the source image is too small for the 2x crop, the augmentation falls back
to a standard 1x crop. The area-average is a simple reshape-and-mean over 2x2 pixel
blocks — no interpolation filter is needed because the 2x factor divides evenly.

### 3. White Balance Application + Perturbation

```
wb = torch.from_numpy(self.wb_multipliers[img_idx]).float()
if self.augment and self.wb_aug_range > 0:
    r_shift = math.exp(rng.uniform(-self.wb_aug_range, self.wb_aug_range))
    b_shift = math.exp(rng.uniform(-self.wb_aug_range, self.wb_aug_range))
    wb = wb * torch.tensor([r_shift, 1.0, b_shift])
rgb = rgb * wb.view(3, 1, 1)
```

**Parameter:** `--wb-aug-range` (default 0.0, typical value 0.15–0.3; requires
`--apply-wb`)

Applies per-image white balance gains and optionally perturbs the R and B channels.
The green channel is held fixed (G=1 normalization). Perturbations are sampled
**uniformly in log space** and then exponentiated, which produces a symmetric
multiplicative distribution. A range of 0.25 in log space corresponds to roughly
±28% gain variation.

Log-space sampling is essential here: a uniform distribution in linear space would be
asymmetric (e.g., halving and doubling are not equidistant from identity). By sampling
`log(shift) ~ Uniform(-r, r)` and applying `exp()`, the augmentation treats a 2x gain
boost and a 0.5x gain reduction as equally likely.

The base WB multipliers come from per-image metadata files (`*_meta.json`) generated
during dataset building. When metadata is missing, identity WB `[1, 1, 1]` is used as
fallback. WB augmentation is gated on `--apply-wb`; without it, `--wb-aug-range` is
silently ignored with a warning.

The `wb` tensor is also reused later for computing per-channel clip levels and
bright-spot amplitude scaling.

### 4. Bright-Spot Augmentation

```
rgb = self._add_bright_spots(rgb, wb, clip_scale, rng)
```

**Parameters:** `--bright-spot-prob` (default 0.0), `--bright-spot-intensity-max`
(default 5.0), `--bright-spot-sigma-max` (default 20.0)

Adds synthetic point light sources (1–5 per patch) that simulate street lights, brake
lights, LEDs, and neon signs. Each spot is an **anisotropic 2D Gaussian blob** with
independent sigma per axis and a random rotation angle, producing elliptical shapes.

Spot colors are sampled from a weighted palette (`_SPOT_PALETTE`) of physically
motivated hues: warm/cool white (tungsten, LED), red (brake lights), amber (turn
signals), blue/cyan/magenta (neon, LEDs). Colors are defined in HSV, converted to
linear RGB via `colorsys.hsv_to_rgb` with a 2.2 gamma linearization. The palette
uses weights to make white spots (the most common real light sources) twice as likely
as colored ones.

The spot amplitude is scaled to `color * wb * intensity`, where `intensity` is sampled
from `[1.5, bright_spot_intensity_max]`. The multiplication by `wb` (but not
`clip_scale`) ensures spots represent real light sources whose brightness is independent
of any highlight augmentation. Spots can push pixel values well above 1.0 before
clipping, creating realistic highlight blowout.

Spot centers are allowed to fall slightly outside the patch boundary (`-0.1*W` to
`1.1*W`) so that edge-feathered partial spots appear naturally. After bright-spot
addition, the CFA image is clamped to per-channel sensor saturation levels during
mosaicing.

### 5. Spatial Flips + 90-Degree Rotations

```
if rng.random() > 0.5:
    rgb = rgb.flip(2)   # horizontal flip
if rng.random() > 0.5:
    rgb = rgb.flip(1)   # vertical flip
k = rng.randint(0, 3)
if k > 0:
    rgb = torch.rot90(rgb, k, [1, 2])
```

Horizontal and vertical flips are applied independently with 50% probability, followed
by a random 90-degree rotation (0, 90, 180, or 270 degrees with equal probability).
This yields all 8 orientations in the dihedral group D4.

All geometric transforms are safe because they are applied to the **RGB data before
mosaicing**. The CFA mask (`self.cfa` / `self.masks`) is pre-computed once and stays
fixed. Since the mosaic operation samples from the transformed RGB using the fixed mask,
the resulting CFA image is physically consistent — it represents what the sensor would
have captured if the scene were in that orientation.

### 6. OLPF Blur Simulation (degradation)

```
sigma = rng.uniform(*self.olpf_sigma)
ks = max(3, int(sigma * 6) | 1)
kernel = _gaussian_kernel_2d(ks, sigma, 3)
rgb = F.conv2d(rgb.unsqueeze(0), kernel, padding=pad, groups=3).squeeze(0)
```

**Parameter:** `--olpf-sigma-max` (default 0.0 = disabled, typical value 0.3–0.8)

Simulates the Optical Low-Pass Filter found in many cameras. The OLPF intentionally
blurs the optical image slightly before it hits the sensor, reducing aliasing from the
CFA subsampling. Many Fuji X-Trans cameras omit the OLPF (relying on the denser
X-Trans pattern for anti-aliasing), but training with simulated OLPF blur improves
robustness when processing images from cameras that do have one, or when optical blur
from lenses varies.

The blur is applied to the full RGB image **before** mosaicing — this is physically
correct because the OLPF acts in the optical domain, before light passes through the
CFA. The kernel size is derived from sigma (`6*sigma`, rounded to the nearest odd
integer, minimum 3), and the Gaussian kernel is generated by `_gaussian_kernel_2d`
from `losses.py`. The convolution is applied per-channel (`groups=3`).

Critically, the ground truth reference `ref` is captured **before** this blur step.
The network therefore learns to recover the sharp original from a blurred-then-mosaiced
input, which encourages sharpness in the reconstruction.

### 7. Sensor Saturation Clipping + Clip Ratio (degradation)

```
clip_levels = wb[self.cfa.long()].unsqueeze(0) * clip_scale  # (1, H, W)
if do_bright_spots:
    cfa_img = cfa_img.clamp(max=clip_levels)

raw_ratio = (cfa_img / (clip_levels + 1e-8)).clamp(0, 1)
clip_ratio = ((raw_ratio - 0.5) * 2.0).clamp(0, 1)  # (1, H, W)
```

After mosaicing, per-pixel clip levels are computed from the WB gains mapped through
the CFA pattern: `clip_levels[y,x] = wb[cfa_channel[y,x]]`. In WB-applied space, the
raw sensor saturation (1.0 in normalized raw) maps to `wb[ch]` per channel, since
`raw_value * wb[ch]` saturates when `raw_value = 1.0`.

If bright spots were added, the CFA image is clamped to these clip levels (simulating
sensor saturation). The **clip ratio** channel then encodes proximity to saturation:
it is 0 for pixels below 50% of their clip level and ramps linearly from 0 to 1
between 50% and 100% of the clip level. This gives the network a soft signal about
where clipping is imminent or has occurred, without encoding absolute luminance.

### 8. Poisson-Gaussian Noise Injection (degradation)

```
read_sigma = rng.uniform(*self.noise_sigma)
shot_coeff = rng.uniform(*self.shot_noise)
if read_sigma > 0 or shot_coeff > 0:
    noise_var = shot_coeff * cfa_img.clamp(min=0) + read_sigma ** 2
    cfa_img = cfa_img + torch.randn_like(cfa_img) * noise_var.sqrt()
```

**Parameters:** `--noise-min` (default 0.0), `--noise-max` (default 0.005),
`--shot-noise-max` (default 0.0)

Noise follows a **Poisson-Gaussian model**: the per-pixel noise standard deviation is
`sqrt(shot_coeff * signal + read_sigma^2)`, where `shot_coeff` controls signal-
dependent shot noise and `read_sigma` controls signal-independent read noise. When
`shot_coeff = 0`, this reduces to pure additive Gaussian noise (the previous behavior).

The noise is injected **after mosaicing**, directly onto the single-channel CFA image.
This matches the physical reality: sensor noise manifests on raw sensor values after the
CFA has already selected which color each pixel records.

Both `read_sigma` and `shot_coeff` are sampled uniformly per patch. Setting the
minimums to 0 ensures some patches are noise-free, preventing the network from learning
a noise-dependent bias. The noise is not clamped after injection — in linear sensor
space, small negative values from noise are physically plausible.

### 9. Five-Channel Input Assembly

```
input_tensor = torch.cat([cfa_img, self.masks, clip_ratio], dim=0)  # (5, H, W)
clip_ch = wb * clip_scale  # (3,) per-channel clip levels for loss
return input_tensor, ref, clip_ch
```

The final input tensor concatenates three components along the channel dimension:

| Channels | Content | Shape |
|---|---|---|
| 0 | Mosaiced CFA image (single-channel sensor data) | `(1, H, W)` |
| 1–3 | Binary CFA channel masks (R, G, B positions) | `(3, H, W)` |
| 4 | Clip ratio (proximity to sensor saturation) | `(1, H, W)` |

The channel masks (`self.masks`) are pre-computed once at dataset construction and
reused for every patch. They tell the network which color filter sits over each pixel.
The clip ratio channel provides a spatially-varying signal about highlight clipping,
enabling the network to adjust its reconstruction strategy near saturation.

The `clip_ch` return value is a 3-element vector of per-channel clip levels (in
WB-applied space), intended for loss masking so the model is not penalized for failing
to reconstruct values above sensor saturation.

### Validation: No Augmentation

The validation dataset is constructed with `augment=False` and `noise_sigma=(0.0, 0.0)`.
This provides a clean, deterministic evaluation signal. Random crops still occur
(for patch extraction), but they remain CFA-aligned and are deterministically seeded
via `_get_rng(idx)` with `rng.seed(idx)`. No flips, rotations, noise, WB perturbation,
bright spots, downscaling, or OLPF blur are applied during validation. The clip ratio
channel and clip levels are still computed (with `wb = ones(3)` if WB is disabled).

## Rationale

| Type | Transform | What It Prevents | Real-World Scenario |
|---|---|---|---|
| Augmentation | CFA-aligned crops | Pattern phase errors in training data | Every sensor pixel has a fixed CFA position |
| Augmentation | Downscale (2x) | Poor performance on lower-resolution content | High-MP sensors, distant subjects, display downsampling |
| Augmentation | Spatial flips + rotations | Orientation bias; limited effective dataset size | Scenes can appear in any orientation |
| Augmentation | WB perturbation | Sensitivity to exact WB gains; color cast artifacts | WB varies by scene illuminant and camera calibration |
| Augmentation | Bright spots | Poor highlight reconstruction around point sources | Street lights, LEDs, brake lights create local clipping |
| Degradation | OLPF blur | Aliasing sensitivity; over-sharpening on soft inputs | Lens blur and OLPF vary across cameras and apertures |
| Degradation | Sensor saturation clipping | Broken highlights; network blind to clipping | Sensor clips at white level; bright sources blow out |
| Degradation | Poisson-Gaussian noise | Overfitting to clean data; artifacts on noisy inputs | All sensors have signal-dependent shot + read noise |
| Feature | Clip ratio channel | Network blind to saturation proximity | Sensor clips at white level; network needs this context |

The augmentation budget is deliberately conservative. Rather than applying many weak
augmentations simultaneously, each augmentation has a clear physical motivation tied
to a specific degree of freedom in the camera imaging pipeline. This avoids creating
unrealistic training distributions that would confuse the network.

## Data Loading Infrastructure

### PatchCacheDataset

`PatchCacheDataset` subclasses `LinearDataset` and replaces direct `.npy` file access
with a streaming patch cache backed by anonymous shared memory (`mmap`).

**Architecture:**

- **Physical buffer:** `N` active slots + `S` staging slots (where
  `S = max(256, int(N * staging_fraction))`). Slot count is derived from
  `--cache-gb` memory budget or defaults to `n_images * patches_per_image`.
- **Slot map indirection:** `slot_map[logical_idx]` maps a logical slot index to a
  physical buffer index. DataLoader workers read through this indirection.
- **Staging separation:** Background streaming threads (`_stream_worker`, 4 threads)
  write **only** to staging slots (physical indices `N..N+S-1`), while DataLoader
  workers read **only** from active slots (physical indices `0..N-1` via `slot_map`).
  This guarantees zero memory contention during training.
- **Pointer swap:** `swap_staging()` atomically updates `slot_map` entries, redirecting
  logical slots to freshly-written physical slots, and returns the old physical slots
  to the free pool. The swap itself is O(swaps) pointer updates (microseconds). It must
  be called when no DataLoader workers are active (between train and val phases).
- **Initial fill:** `_fill_buffer()` populates all `N` active slots at construction
  time using a thread pool (8 workers).

When `downscale_prob > 0`, patches are extracted at `2 * patch_size` and the downscale
decision is deferred to `__getitem__`. If downscale is not chosen, a random CFA-aligned
sub-crop of the 2x patch is taken instead.

The shared memory is allocated via `mmap.mmap(-1, nbytes, MAP_SHARED | MAP_ANONYMOUS)`
so that forked DataLoader workers see the same physical pages without pickling.

### ImageGroupedSampler

`ImageGroupedSampler` is a `torch.utils.data.Sampler` that yields patch indices grouped
by source image. Instead of fully shuffling all `N * patches_per_image` indices (which
scatters patches from the same image across workers and causes redundant file opens),
it:

1. Shuffles at the **image level** (seeded by `self.epoch` for reproducibility)
2. Emits all `patches_per_image` patches for each image consecutively
3. Shuffles the patch order within each image group

Combined with `LinearDataset._load_image` (which keeps a per-worker mmap cache of
size 1), this reduces file opens from `N * patches_per_image` down to `N` per epoch.
The sampler exposes `set_epoch(epoch)` for distributed training reproducibility.

## Key Files

| File | Role |
|---|---|
| `dataset.py` | `LinearDataset._process_patch` — all augmentation logic; `__getitem__` handles crop + downscale |
| `dataset.py` | `PatchCacheDataset` — streaming patch cache with pointer-swap; subclasses `LinearDataset` |
| `dataset.py` | `ImageGroupedSampler` — cache-friendly sampler that groups patches by source image |
| `dataset.py` | `create_mixed_dataset()` — passes augmentation params, adds torture patterns |
| `train.py` | CLI arguments: `--noise-min`, `--noise-max`, `--shot-noise-max`, `--olpf-sigma-max`, `--wb-aug-range`, `--bright-spot-prob`, `--downscale-prob` |
| `cfa.py` | `cfa_period()`, `patch_alignment()` — alignment constraints for crops and patch sizes |
| `losses.py` | `_gaussian_kernel_2d()` — shared Gaussian kernel used by OLPF simulation |

## Antipatterns

### Note on 90-Degree Rotations

Earlier versions of this codebase forbade 90-degree rotations because they transpose
the spatial axes and would break CFA alignment **if applied after mosaicing or with a
fixed CFA mask applied to the rotated data**. The current code applies `torch.rot90`
to the RGB image **before** mosaicing — the CFA mask is then applied fresh to the
already-rotated RGB, so the rotation is safe. The legacy restriction documented in
`src/datasets/dataset_v4.py` no longer applies.

### Wrong Crop Alignment

Cropping at an offset that is not a multiple of the CFA period creates a phase mismatch
between the pre-computed CFA mask and the actual color filter positions in the patch.
For X-Trans (period 6), a crop starting at row 3 would shift the pattern by half a
tile. The network would see inconsistent CFA-to-color mappings across different patches
from the same image, making learning impossible.

The alignment constraint `top = (rand // period) * period` is the fix. Any change to
the crop logic must preserve this quantization.

### Noise Before Mosaicing

Injecting noise onto the full RGB image before mosaicing would produce spatially
correlated noise across CFA neighbors — unlike real sensor noise, which is independent
per photosite. The current code correctly adds noise **after** `mosaic()`, onto the
single-channel CFA image.

### OLPF After Mosaicing

Applying Gaussian blur after mosaicing would blur across CFA color boundaries, mixing
R, G, and B sensor values — something that never happens optically. The OLPF
simulation correctly blurs the RGB image before mosaicing, in the optical domain.

### Legacy Warning

`src/datasets/dataset_v4.py` is a **legacy dataset implementation**. It lacks WB
augmentation, bright-spot augmentation, Poisson-Gaussian noise, clip ratio computation,
OLPF simulation, downscale augmentation, and multi-CFA support. Do not use it for new
training runs. The current implementation in `dataset.py` (root) supersedes it
entirely.
