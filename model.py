# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
"""
U-Net for X-Trans demosaicing.

Architecture: encoder-decoder with skip connections.
- Input: 5 channels (CFA + position masks + clip ratio)
- Output: 3 channels (RGB)
- Additive residual: output = CFA_per_channel + learned_delta
- 4 levels: 64 -> 128 -> 256 -> 512
- 3x3 convolutions throughout
- Receptive field easily covers 2-3 X-Trans repeats (12-18 pixels)
"""

import math

import torch
import torch.nn as nn



class ConvBlock(nn.Module):
    """Two convolutions with LayerNorm and ReLU. First kernel size is configurable."""

    def __init__(self, in_ch: int, out_ch: int, first_kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, first_kernel, padding=first_kernel // 2),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample with strided convolution then ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    """Upsample with PixelShuffle, concatenate skip, then ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 1),
            nn.PixelShuffle(2),
        )
        self.conv = ConvBlock(out_ch * 2, out_ch)  # *2 for skip concat

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class XTransUNet(nn.Module):
    """
    U-Net for X-Trans demosaicing.

    4 encoder levels, 4 decoder levels, skip connections at each level.
    Channel widths: base_width * [1, 2, 4, 8, 16] (default 64 → 64..1024).
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 3,
                 base_width: int = 64, cfa_period: int = 2):
        super().__init__()
        w = base_width
        self.cfa_period = cfa_period

        # Positional encoding channels: sin/cos for row and column phase
        pos_channels = 4 if cfa_period > 2 else 0
        stem_kernel = 7 if cfa_period > 2 else 3

        # Encoder
        self.enc1 = ConvBlock(in_channels + pos_channels, w, first_kernel=stem_kernel)
        self.enc2 = DownBlock(w, w * 2)
        self.enc3 = DownBlock(w * 2, w * 4)
        self.enc4 = DownBlock(w * 4, w * 8)

        # Bottleneck
        self.bottleneck = DownBlock(w * 8, w * 16)

        # Decoder
        self.dec4 = UpBlock(w * 16, w * 8)
        self.dec3 = UpBlock(w * 8, w * 4)
        self.dec2 = UpBlock(w * 4, w * 2)
        self.dec1 = UpBlock(w * 2, w)

        # Output
        self.out_conv = nn.Conv2d(w, out_channels, 1)

    def forward(self, x):
        cfa = x[:, 0:1]    # (B, 1, H, W)
        masks = x[:, 1:4]  # (B, 3, H, W) — R, G, B position masks
        baseline = cfa * masks  # (B, 3, H, W) — value only in its true channel

        # CFA periodic positional encoding for non-Bayer patterns
        if self.cfa_period > 2:
            B, _, H, W = x.shape
            y = torch.arange(H, device=x.device, dtype=x.dtype).unsqueeze(1).expand(H, W)
            xc = torch.arange(W, device=x.device, dtype=x.dtype).unsqueeze(0).expand(H, W)
            phase = 2 * math.pi / self.cfa_period
            pos_enc = torch.stack([
                torch.sin(phase * y),
                torch.cos(phase * y),
                torch.sin(phase * xc),
                torch.cos(phase * xc),
            ]).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 4, H, W)
            x = torch.cat([x, pos_enc], dim=1)

        # Encoder
        e1 = self.enc1(x)   # 64, H, W
        e2 = self.enc2(e1)  # 128, H/2, W/2
        e3 = self.enc3(e2)  # 256, H/4, W/4
        e4 = self.enc4(e3)  # 512, H/8, W/8

        # Bottleneck
        b = self.bottleneck(e4)  # 1024, H/16, W/16

        # Decoder with skip connections
        d4 = self.dec4(b, e4)   # 512, H/8, W/8
        d3 = self.dec3(d4, e3)  # 256, H/4, W/4
        d2 = self.dec2(d3, e2)  # 128, H/2, W/2
        d1 = self.dec1(d2, e1)  # 64, H, W

        return baseline + self.out_conv(d1)  # 3, H, W


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys

    base_width = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    model = XTransUNet(base_width=base_width)
    print(f"base_width={base_width}, Parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(1, 5, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
