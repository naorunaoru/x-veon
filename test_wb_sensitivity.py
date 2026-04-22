#!/usr/bin/env python3
"""
Quick WB mask sensitivity tests.

1. Gradient test (random weights): does WB gradient flow to the loss?
2. Output sensitivity (random weights): how much does output change when WB changes?
3. Ablation (trained checkpoint): real WB vs identity WB on validation data.

Usage:
    # Tests 1-2 (no checkpoint needed)
    python test_wb_sensitivity.py

    # Test 3: ablation on a trained checkpoint
    python test_wb_sensitivity.py --checkpoint path/to/best.pt --data-dir /path/to/val_npy
"""

import argparse
import torch
import numpy as np

from cfa import CFA_REGISTRY, cfa_period as _cfa_period
from model import XTransUNet


def gradient_test(cfa_type: str = "xtrans", base_width: int = 16):
    """Check that gradients flow from loss back to WB input."""
    pattern = CFA_REGISTRY[cfa_type]
    cp = _cfa_period(pattern)
    model = XTransUNet(base_width=base_width, cfa_period=cp,
                       cfa_pattern=torch.from_numpy(pattern))
    model.train()

    x = torch.randn(2, 1, 96, 96)
    wb = torch.tensor([[2.1, 1.0, 1.5], [1.8, 1.0, 2.0]], requires_grad=True)
    target = torch.randn(2, 3, 96, 96)

    out = model(x, wb)
    loss = (out - target).pow(2).mean()
    loss.backward()

    grad_norm = wb.grad.norm().item()
    print(f"[gradient] cfa={cfa_type} | WB grad norm: {grad_norm:.6f}")
    print(f"           WB grad per element: {wb.grad}")
    assert grad_norm > 0, "WB gradient is zero — mask is disconnected!"
    return grad_norm


def output_sensitivity(cfa_type: str = "xtrans", base_width: int = 16):
    """Measure output change when WB changes, holding input constant."""
    pattern = CFA_REGISTRY[cfa_type]
    cp = _cfa_period(pattern)
    model = XTransUNet(base_width=base_width, cfa_period=cp,
                       cfa_pattern=torch.from_numpy(pattern))
    model.eval()

    x = torch.randn(1, 1, 96, 96)
    wb_real = torch.tensor([[2.1, 1.0, 1.5]])
    wb_identity = torch.tensor([[1.0, 1.0, 1.0]])

    with torch.no_grad():
        out_real = model(x, wb_real)
        out_identity = model(x, wb_identity)

    diff = (out_real - out_identity).abs()
    print(f"[sensitivity] cfa={cfa_type}")
    print(f"  WB=[2.1, 1.0, 1.5] vs [1.0, 1.0, 1.0]:")
    print(f"  mean diff: {diff.mean():.6f}")
    print(f"  max diff:  {diff.max():.6f}")
    print(f"  (random weights — confirms mask is wired correctly)")
    return diff.mean().item()


def stem_weight_analysis(checkpoint_path: str):
    """After training: check if the enc1 WB channel weights are alive or dead."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]

    # enc1.block.0 is the first conv: its weights are (out_ch, in_ch, kH, kW)
    w = state["enc1.block.0.weight"]
    out_ch, in_ch, kh, kw = w.shape

    # Channel layout: [CFA(0), R_mask(1), G_mask(2), B_mask(3), WB_mask(4), ...]
    wb_idx = 4
    if in_ch <= wb_idx:
        print(f"[stem] Checkpoint has {in_ch} input channels — no WB channel present")
        return

    wb_weights = w[:, wb_idx]
    other_weights = torch.cat([w[:, :wb_idx], w[:, wb_idx+1:]], dim=1)

    wb_norm = wb_weights.norm().item()
    other_norm = other_weights.norm().item() / (in_ch - 1)  # per-channel average

    ratio = wb_norm / (other_norm + 1e-8)
    print(f"[stem] WB channel weight norm:     {wb_norm:.4f}")
    print(f"[stem] Other channels avg norm:    {other_norm:.4f}")
    print(f"[stem] Ratio (WB / avg other):     {ratio:.4f}")
    if ratio < 0.1:
        print(f"  → WB channel is nearly dead — model likely ignores it")
    elif ratio < 0.5:
        print(f"  → WB channel is weak — model uses it modestly")
    else:
        print(f"  → WB channel is active — model learned to use it")


def ablation(checkpoint_path: str, data_dir: str, cfa_type: str = "xtrans",
             base_width: int | None = None, n_patches: int = 200):
    """Compare PSNR with real WB vs identity WB on validation data."""
    from dataset import LinearDataset

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bw = base_width or ckpt.get("base_width", 64)
    ct = ckpt.get("cfa_type", cfa_type)
    pattern = CFA_REGISTRY[ct]
    cp = _cfa_period(pattern)

    model = XTransUNet(base_width=bw, cfa_period=cp,
                       cfa_pattern=torch.from_numpy(pattern))
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ds = LinearDataset(data_dir=data_dir, patch_size=96, augment=False,
                       noise_sigma=(0, 0), shot_noise=(0, 0), cfa_type=ct,
                       patches_per_image=1)

    n = min(n_patches, len(ds))
    psnr_real = []
    psnr_identity = []

    with torch.no_grad():
        for i in range(n):
            cfa_img, ref, wb = ds[i]
            cfa_img = cfa_img.unsqueeze(0).to(device)
            ref = ref.to(device)
            wb_real = wb.unsqueeze(0).to(device)
            wb_id = torch.ones(1, 3, device=device)

            out_real = model(cfa_img, wb_real)[0]
            out_id = model(cfa_img, wb_id)[0]

            mse_real = (out_real - ref).pow(2).mean().item()
            mse_id = (out_id - ref).pow(2).mean().item()

            psnr_real.append(-10 * np.log10(mse_real + 1e-10))
            psnr_identity.append(-10 * np.log10(mse_id + 1e-10))

    psnr_real = np.mean(psnr_real)
    psnr_identity = np.mean(psnr_identity)
    delta = psnr_real - psnr_identity

    print(f"\n[ablation] {n} patches from {data_dir}")
    print(f"  PSNR (real WB):     {psnr_real:.3f} dB")
    print(f"  PSNR (identity WB): {psnr_identity:.3f} dB")
    print(f"  Delta:              {delta:+.3f} dB")
    if abs(delta) < 0.01:
        print(f"  → Model ignores WB mask (no measurable difference)")
    elif delta > 0:
        print(f"  → Model benefits from WB mask (+{delta:.3f} dB)")
    else:
        print(f"  → WB mask hurts?! (investigate — likely a bug)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--cfa-type", type=str, default="xtrans")
    parser.add_argument("--base-width", type=int, default=None)
    parser.add_argument("--n-patches", type=int, default=200)
    args = parser.parse_args()

    print("=" * 60)
    print("WB MASK SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Always run architecture tests
    for ct in ["bayer", "xtrans"]:
        gradient_test(ct)
    print()
    for ct in ["bayer", "xtrans"]:
        output_sensitivity(ct)

    # Trained checkpoint analysis
    if args.checkpoint:
        print()
        stem_weight_analysis(args.checkpoint)

        if args.data_dir:
            print()
            ablation(args.checkpoint, args.data_dir,
                     cfa_type=args.cfa_type,
                     base_width=args.base_width,
                     n_patches=args.n_patches)
        else:
            print("\n(Pass --data-dir to run the ablation test)")
