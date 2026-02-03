#!/usr/bin/env python3
"""Fit a color correction matrix by comparing rawpy raw vs sRGB outputs."""
import rawpy
import numpy as np
from PIL import Image

raw = rawpy.imread("test_rafs/DSCF5212.RAF")

# rawpy in camera colorspace (no color matrix applied)
ref_raw = raw.postprocess(
    use_camera_wb=True,
    output_color=rawpy.ColorSpace.raw,
    output_bps=8,
    no_auto_bright=True,
).astype(np.float64) / 255.0

# rawpy with full sRGB conversion
ref_srgb = raw.postprocess(
    use_camera_wb=True,
    output_color=rawpy.ColorSpace.sRGB,
    output_bps=8,
    no_auto_bright=True,
).astype(np.float64) / 255.0

print(f"Raw colorspace mean: R={ref_raw[:,:,0].mean():.3f} G={ref_raw[:,:,1].mean():.3f} B={ref_raw[:,:,2].mean():.3f}")
print(f"sRGB mean:           R={ref_srgb[:,:,0].mean():.3f} G={ref_srgb[:,:,1].mean():.3f} B={ref_srgb[:,:,2].mean():.3f}")

# Flatten and subsample
N = ref_raw.shape[0] * ref_raw.shape[1]
raw_flat = ref_raw.reshape(N, 3)
srgb_flat = ref_srgb.reshape(N, 3)

idx = np.random.default_rng(42).choice(N, min(200000, N), replace=False)
raw_sub = raw_flat[idx]
srgb_sub = srgb_flat[idx]

# Fit: srgb = raw @ M_T  =>  M_T = lstsq(raw, srgb)
M_T, residuals, rank, sv = np.linalg.lstsq(raw_sub, srgb_sub, rcond=None)
M = M_T.T

print(f"\nFitted 3x3 matrix (camera_gamma -> sRGB_gamma):")
for i, name in enumerate("RGB"):
    print(f"  {name}: [{M[i,0]:+.4f} {M[i,1]:+.4f} {M[i,2]:+.4f}]  sum={M[i].sum():.4f}")

# Check fit quality
pred = raw_sub @ M_T
err = np.abs(pred - srgb_sub).mean() * 255
print(f"\nMean abs error: {err:.2f} / 255")
max_err = np.abs(pred - srgb_sub).max() * 255
print(f"Max abs error:  {max_err:.2f} / 255")

# Now let's also do this in linear space
def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1/2.4) - 0.055)

raw_lin = srgb_to_linear(raw_sub)
srgb_lin = srgb_to_linear(srgb_sub)

M_T_lin, _, _, _ = np.linalg.lstsq(raw_lin, srgb_lin, rcond=None)
M_lin = M_T_lin.T

print(f"\nFitted 3x3 matrix (camera_linear -> sRGB_linear):")
for i, name in enumerate("RGB"):
    print(f"  {name}: [{M_lin[i,0]:+.4f} {M_lin[i,1]:+.4f} {M_lin[i,2]:+.4f}]  sum={M_lin[i].sum():.4f}")

pred_lin = raw_lin @ M_T_lin
pred_gamma = linear_to_srgb(np.clip(pred_lin, 0, None))
err_lin = np.abs(pred_gamma - srgb_sub).mean() * 255
print(f"Mean abs error (linear fit): {err_lin:.2f} / 255")

# Save the linear matrix as it's more principled
print(f"\nRecommended matrix (apply in linear space):")
print(repr(M_lin))

# Also test on a second file (X-T10)
print("\n--- X-T10 test ---")
raw2 = rawpy.imread("test_rafs/_DSF4185.RAF")
ref_raw2 = raw2.postprocess(
    use_camera_wb=True, output_color=rawpy.ColorSpace.raw,
    output_bps=8, no_auto_bright=True,
).astype(np.float64) / 255.0
ref_srgb2 = raw2.postprocess(
    use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB,
    output_bps=8, no_auto_bright=True,
).astype(np.float64) / 255.0

N2 = ref_raw2.shape[0] * ref_raw2.shape[1]
idx2 = np.random.default_rng(42).choice(N2, min(200000, N2), replace=False)
raw2_lin = srgb_to_linear(ref_raw2.reshape(N2, 3)[idx2])
srgb2_lin = srgb_to_linear(ref_srgb2.reshape(N2, 3)[idx2])

M_T2, _, _, _ = np.linalg.lstsq(raw2_lin, srgb2_lin, rcond=None)
M2 = M_T2.T
print(f"X-T10 matrix (camera_linear -> sRGB_linear):")
for i, name in enumerate("RGB"):
    print(f"  {name}: [{M2[i,0]:+.4f} {M2[i,1]:+.4f} {M2[i,2]:+.4f}]  sum={M2[i].sum():.4f}")
