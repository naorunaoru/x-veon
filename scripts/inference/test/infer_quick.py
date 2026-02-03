import torch
import rawpy
import numpy as np
from model import XTransUNet
from PIL import Image
import sys

device = torch.device("mps")
ckpt = torch.load(sys.argv[1], map_location=device, weights_only=False)
epoch = ckpt.get("epoch", "?")
psnr = ckpt.get("best_val_psnr", 0)
print(f"Loaded: epoch {epoch}, psnr {psnr:.2f} dB")

model = XTransUNet().to(device)
model.load_state_dict(ckpt["model"])
model.eval()

raw = rawpy.imread(sys.argv[2])
cfa = raw.raw_image_visible.astype(np.float32)
black = np.array(raw.black_level_per_channel).mean()
white = raw.white_level
cfa = (cfa - black) / (white - black)
cfa = np.clip(cfa, 0, 1)
h, w = cfa.shape
print(f"CFA: {w}x{h}")

ph = (6 - h % 6) % 6
pw = (6 - w % 6) % 6
if ph or pw:
    cfa = np.pad(cfa, ((0, ph), (0, pw)), mode="reflect")
    h, w = cfa.shape

patch_size = 288
overlap = 48
step = patch_size - overlap
out = np.zeros((h, w, 3), dtype=np.float32)
weights = np.zeros((h, w, 1), dtype=np.float32)
hann_1d = np.hanning(patch_size)
hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)

with torch.no_grad():
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patch = cfa[y:y+patch_size, x:x+patch_size]
            inp = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(inp).squeeze(0).permute(1, 2, 0).cpu().numpy()
            out[y:y+patch_size, x:x+patch_size] += pred * hann_2d[:,:,None]
            weights[y:y+patch_size, x:x+patch_size] += hann_2d[:,:,None]
        print(f"Row {y}/{h}", end="\r")

out = out / np.maximum(weights, 1e-8)
print("\nBlending done")

oh, ow = raw.raw_image_visible.shape
out = out[:oh, :ow]

wb = np.array(raw.camera_whitebalance[:3])
wb = wb / wb[1]
out = out * wb
out = np.clip(out, 0, 1)
out = np.power(out, 1/2.2)

out_u8 = (out * 255).astype(np.uint8)
Image.fromarray(out_u8).save(sys.argv[3])
print(f"Saved {sys.argv[3]}")
