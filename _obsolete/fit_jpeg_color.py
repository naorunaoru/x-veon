#!/usr/bin/env python3
"""
Fit a color correction matrix by comparing model output (no color correction)
with the camera's own JPEG output. This captures the camera's full color
pipeline including film simulation (Provia etc).
"""
import numpy as np
from PIL import Image
import os
import json

def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.clip(x, 0.0031308, None), 1/2.4) - 0.055)

# Map test RAF filenames to their JPEG folder paths
jpeg_base = "data/fuji_jpgs"
raf_to_jpeg = {}
for folder in os.listdir(jpeg_base):
    folder_path = os.path.join(jpeg_base, folder)
    if not os.path.isdir(folder_path):
        continue
    for f in os.listdir(folder_path):
        if f.upper().endswith('.JPG'):
            stem = os.path.splitext(f)[0]
            raf_to_jpeg[stem] = os.path.join(folder_path, f)

# Collect pixel pairs from model output vs camera JPEG
all_model = []
all_jpeg = []

output_dir = "output_nocorr"
if not os.path.exists(output_dir):
    print(f"ERROR: Run inference first with --no-color-correction --output-dir {output_dir}")
    exit(1)

for f in sorted(os.listdir(output_dir)):
    if not f.endswith('_demosaic.png'):
        continue
    stem = f.replace('_demosaic.png', '')
    
    if stem not in raf_to_jpeg:
        print(f"  {stem}: no matching JPEG, skipping")
        continue
    
    # Load model output
    model_img = np.array(Image.open(os.path.join(output_dir, f))).astype(np.float64) / 255.0
    
    # Load camera JPEG (may be different size due to RAF vs JPEG crop)
    jpeg_img = np.array(Image.open(raf_to_jpeg[stem])).astype(np.float64) / 255.0
    
    # Resize JPEG to match model output dimensions
    if model_img.shape[:2] != jpeg_img.shape[:2]:
        jpeg_pil = Image.open(raf_to_jpeg[stem])
        jpeg_pil = jpeg_pil.resize((model_img.shape[1], model_img.shape[0]), Image.LANCZOS)
        jpeg_img = np.array(jpeg_pil).astype(np.float64) / 255.0
    
    print(f"  {stem}: model {model_img.shape} vs jpeg {jpeg_img.shape}")
    
    # Subsample pixels
    N = model_img.shape[0] * model_img.shape[1]
    rng = np.random.default_rng(42)
    idx = rng.choice(N, min(50000, N), replace=False)
    
    all_model.append(model_img.reshape(N, 3)[idx])
    all_jpeg.append(jpeg_img.reshape(N, 3)[idx])

if not all_model:
    print("No matching pairs found!")
    exit(1)

model_pixels = np.vstack(all_model)
jpeg_pixels = np.vstack(all_jpeg)
print(f"\nTotal pixel pairs: {len(model_pixels)}")

# Fit in gamma space
M_T_gamma, _, _, _ = np.linalg.lstsq(model_pixels, jpeg_pixels, rcond=None)
M_gamma = M_T_gamma.T
print(f"\nFitted matrix (gamma space, model -> camera JPEG):")
for i, name in enumerate("RGB"):
    print(f"  {name}: [{M_gamma[i,0]:+.4f} {M_gamma[i,1]:+.4f} {M_gamma[i,2]:+.4f}]  sum={M_gamma[i].sum():.4f}")

pred_gamma = model_pixels @ M_T_gamma
err_gamma = np.abs(pred_gamma - jpeg_pixels).mean() * 255
print(f"Mean abs error (gamma): {err_gamma:.2f} / 255")

# Fit in linear space
model_lin = srgb_to_linear(model_pixels)
jpeg_lin = srgb_to_linear(jpeg_pixels)

M_T_lin, _, _, _ = np.linalg.lstsq(model_lin, jpeg_lin, rcond=None)
M_lin = M_T_lin.T
print(f"\nFitted matrix (linear space, model -> camera JPEG):")
for i, name in enumerate("RGB"):
    print(f"  {name}: [{M_lin[i,0]:+.4f} {M_lin[i,1]:+.4f} {M_lin[i,2]:+.4f}]  sum={M_lin[i].sum():.4f}")

pred_lin = linear_to_srgb(np.clip(model_lin @ M_T_lin, 0, None))
err_lin = np.abs(pred_lin - jpeg_pixels).mean() * 255
print(f"Mean abs error (linear): {err_lin:.2f} / 255")

# Save the better matrix
print(f"\n--- Best matrix ---")
if err_gamma < err_lin:
    print("Gamma-space fit is better")
    best = M_gamma
    space = "gamma"
else:
    print("Linear-space fit is better")
    best = M_lin
    space = "linear"

result = {
    "matrix": best.tolist(),
    "space": space,
    "error_per_255": float(min(err_gamma, err_lin)),
    "n_images": len(all_model),
    "n_pixels": len(model_pixels),
}
with open("color_correction_matrix.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to color_correction_matrix.json")
print(f"\nMatrix:\n{repr(best)}")
