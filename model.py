# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
"""
U-Net for CFA demosaicing (X-Trans / Bayer).

Architecture: encoder-decoder with skip connections.
- Input: 1 channel (CFA mosaic)
- Output: 3 channels (RGB)
- CFA channel masks and WB mask are generated internally from stored buffers
- Additive residual: output = CFA_per_channel + learned_delta
- 4 levels: base_width * [1, 2, 4, 8, 16]
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

    def __init__(self, in_ch: int, out_ch: int, fp32_up: bool = False):
        super().__init__()
        self.fp32_up = fp32_up
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 1),
            nn.PixelShuffle(2),
        )
        self.conv = ConvBlock(out_ch * 2, out_ch)  # *2 for skip concat

    def forward(self, x, skip):
        if self.fp32_up:
            orig_dtype = x.dtype
            x = self.up[0](x.float()).to(orig_dtype)
            x = self.up[1](x)
        else:
            x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class XTransUNet(nn.Module):
    """
    U-Net for X-Trans demosaicing.

    4 encoder levels, 4 decoder levels, skip connections at each level.
    Channel widths: base_width * [1, 2, 4, 8, 16] (default 64 → 64..1024).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 base_width: int = 64, cfa_period: int = 2,
                 cfa_pattern: torch.Tensor | None = None):
        super().__init__()
        w = base_width
        self.cfa_period = cfa_period

        # Store CFA pattern as buffer (travels with .to(device), saved in state_dict)
        if cfa_pattern is None:
            cfa_pattern = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        self.register_buffer('cfa_pattern', cfa_pattern.long())

        # Tiled pattern is cached by (H, W) to avoid recomputing every call
        self._cached_tiled: torch.Tensor | None = None
        self._cached_hw: tuple[int, int] = (0, 0)

        # Positional encoding channels: sin/cos for row and column phase
        # Mask channels (3) + WB mask (1) are generated internally and concatenated with input
        pos_channels = 4 if cfa_period > 2 else 0
        stem_kernel = 7 if cfa_period > 2 else 3

        # Encoder: 1 (CFA) + 3 (masks) + 1 (WB mask) + pos_channels
        self.enc1 = ConvBlock(in_channels + 4 + pos_channels, w, first_kernel=stem_kernel)
        self.enc2 = DownBlock(w, w * 2)
        self.enc3 = DownBlock(w * 2, w * 4)
        self.enc4 = DownBlock(w * 4, w * 8)

        # Bottleneck
        self.bottleneck = DownBlock(w * 8, w * 16)

        # Decoder
        self.dec4 = UpBlock(w * 16, w * 8, fp32_up=True)
        self.dec3 = UpBlock(w * 8, w * 4)
        self.dec2 = UpBlock(w * 4, w * 2)
        self.dec1 = UpBlock(w * 2, w)

        # Output
        self.out_conv = nn.Conv2d(w, out_channels, 1)

    def _tile_pattern(self, H: int, W: int) -> torch.Tensor:
        """Tile CFA pattern to (H, W). Cached by (H, W)."""
        if self._cached_hw == (H, W) and self._cached_tiled is not None:
            return self._cached_tiled
        ph, pw = self.cfa_pattern.shape
        tiled = self.cfa_pattern.repeat((H + ph - 1) // ph, (W + pw - 1) // pw)[:H, :W]
        self._cached_tiled = tiled
        self._cached_hw = (H, W)
        return tiled

    def _make_masks(self, H: int, W: int) -> torch.Tensor:
        """Tile CFA pattern to (1, 3, H, W) channel masks."""
        tiled = self._tile_pattern(H, W)
        masks = torch.zeros(1, 3, H, W, device=self.cfa_pattern.device, dtype=torch.float32)
        masks[0, 0] = (tiled == 0).float()  # R
        masks[0, 1] = (tiled == 1).float()  # G
        masks[0, 2] = (tiled == 2).float()  # B
        return masks

    def _make_wb_mask(self, H: int, W: int, wb: torch.Tensor) -> torch.Tensor:
        """Build (B, 1, H, W) WB mask: each pixel gets its CFA channel's WB coefficient."""
        tiled = self._tile_pattern(H, W)  # (H, W) with values 0/1/2
        # wb: (B, 3) → gather per-pixel WB value via the CFA pattern
        return wb[:, tiled.flatten()].view(wb.shape[0], 1, H, W)

    def forward(self, x, wb: torch.Tensor):
        """
        Args:
            x:  (B, 1, H, W) CFA mosaic
            wb: (B, 3) white balance coefficients (R, G, B)
        """
        B, _, H, W = x.shape
        cfa = x[:, 0:1]    # (B, 1, H, W)
        masks = self._make_masks(H, W)  # (1, 3, H, W) — broadcasts over batch
        baseline = cfa * masks  # (B, 3, H, W) — value only in its true channel

        # Normalize WB to green=1.0 (idempotent if already normalized)
        wb = wb / wb[:, 1:2]
        wb_mask = self._make_wb_mask(H, W, wb)  # (B, 1, H, W)

        # Concatenate CFA + channel masks + WB mask for encoder input
        x = torch.cat([x, masks.expand(B, -1, -1, -1), wb_mask], dim=1)  # (B, 5, H, W)

        # CFA periodic positional encoding for non-Bayer patterns
        if self.cfa_period > 2:
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
    from cfa import CFA_REGISTRY, cfa_period as _cfa_period

    base_width = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    cfa_type = sys.argv[2] if len(sys.argv) > 2 else "bayer"
    pattern = CFA_REGISTRY[cfa_type]
    cp = _cfa_period(pattern)
    model = XTransUNet(base_width=base_width, cfa_period=cp,
                       cfa_pattern=torch.from_numpy(pattern))
    print(f"base_width={base_width}, cfa={cfa_type}, Parameters: {count_parameters(model):,}")

    # Test forward pass — input is 1 channel (CFA mosaic), wb is (B, 3)
    x = torch.randn(1, 1, 256, 256)
    wb = torch.tensor([[2.1, 1.0, 1.5]])  # example WB coefficients
    y = model(x, wb)
    print(f"Input:  {x.shape}")
    print(f"WB:     {wb.shape}")
    print(f"Output: {y.shape}")
