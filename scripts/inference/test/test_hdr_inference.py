#!/usr/bin/env python3
"""Test HDR HEIC output with model inference."""

import numpy as np
import torch
import rawpy
from model import XTransUNet
from xtrans_pattern import make_channel_masks
from hdr_heic import save_hdr_heic

# Load model
device = torch.device("mps")
ckpt = torch.load("checkpoints_v4_ft/best.pt", map_location=device, weights_only=False)
epoch = ckpt.get('epoch', '?')
psnr = ckpt.get('best_val_psnr', 0)
print(f"Checkpoint: epoch {epoch}, PSNR {psnr:.1f} dB")

model = XTransUNet().to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# Load RAF
raw = rawpy.imread("test_rafs/DSCF3561.RAF")
cam_to_xyz = raw.rgb_xyz_matrix[:3, :3]
print(f"Camera to XYZ matrix:\n{cam_to_xyz}")

# Get linear CFA
cfa = raw.raw_image_visible.astype(np.float32)
black = np.array(raw.black_level_per_channel).mean()
white = raw.white_level
cfa = np.clip((cfa - black) / (white - black), 0, 1).astype(np.float32)
h, w = cfa.shape

# Crop to small test region (faster)
crop_h, crop_w = 960, 960
start_y = (h - crop_h) // 2
start_x = (w - crop_w) // 2
cfa_crop = cfa[start_y:start_y+crop_h, start_x:start_x+crop_w]
print(f"Test crop: {crop_w}x{crop_h}")

# Prepare input with channel masks
channel_masks = make_channel_masks(crop_h, crop_w)
cfa_tensor = torch.from_numpy(cfa_crop).float().unsqueeze(0)
input_tensor = torch.cat([cfa_tensor, channel_masks], dim=0).unsqueeze(0).to(device)

# Inference
print("Running inference...")
with torch.no_grad():
    output = model(input_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
print(f"Model output range: [{output.min():.4f}, {output.max():.4f}]")

# Apply white balance
wb = np.array(raw.camera_whitebalance[:3])
wb = wb / wb[1]
output_wb = output * wb
print(f"After WB range: [{output_wb.min():.4f}, {output_wb.max():.4f}]")

# Save as HDR HEIC
save_hdr_heic(output_wb, "test_crop_hdr.heic", cam_to_xyz, peak_nits=1000)
print("Done!")
