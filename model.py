# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
"""
Packed demosaicing network (architecture v7).

- Input: 5 channels (mosaic, R/G/B position masks, clip ratio), as the app sends them.
- The mosaic is divided by its mean before the network and the result multiplied back,
  so a tile's exposure does not change what the network sees.
- The input is packed by space-to-depth (3x3 for X-Trans, 2x2 for Bayer): every channel
  is one photosite position, and the network works at 1/f and coarser resolutions.
- Output: 3 channels (RGB) = measured samples in their own channels + learned correction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHITECTURE_TAG = "v7"

# Tiles whose mean is below this are processed at a fixed scale. A numerical-stability
# choice: a black or noisy near-black tile can have a mean of zero or less.
MEAN_FLOOR = 1e-4


def packed_channels(base_width: int, cfa_period: int) -> int:
    """Channels at the packed resolution: 72 (X-Trans) and 44 (Bayer) for base_width 16."""
    return base_width * 9 // 2 if cfa_period > 2 else base_width * 11 // 4


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 convolutions with ReLU. No normalisation layers."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class XTransUNet(nn.Module):
    """Packed U-shaped network for X-Trans (cfa_period=6) and Bayer (cfa_period=2).

    `stages` counts resolution reductions, the packing being the first: S uses 2
    (1/3 and 1/6 for X-Trans, 1/2 and 1/4 for Bayer).
    """

    def __init__(self, base_width: int = 16, cfa_period: int = 2, stages: int = 2,
                 gain_jitter_stops: float = 0.0) -> None:
        super().__init__()
        if stages < 2:
            raise ValueError("stages must be at least 2 (the packing and one halving)")
        self.cfa_period = cfa_period
        self.stages = stages
        self.gain_jitter_stops = gain_jitter_stops
        self.factor = 3 if cfa_period > 2 else 2
        # X-Trans 3x3 cells come in two types (red and blue swapped); the packed masks
        # tell the network which one it is in. Every Bayer cell is the same.
        self.pack_masks = cfa_period > 2

        c = packed_channels(base_width, cfa_period)
        self.enc = _block((5 if self.pack_masks else 2) * self.factor ** 2, c)
        self.downs = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        ch = c
        for _ in range(stages - 1):
            self.downs.append(nn.Conv2d(ch, ch, 2, stride=2))
            self.down_blocks.append(_block(ch, ch * 2))
            ch *= 2
        for _ in range(stages - 1):
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1), nn.PixelShuffle(2)))
            self.up_blocks.append(_block(ch, ch // 2))
            ch //= 2
        self.head = nn.Conv2d(c, 3 * self.factor ** 2, 1)

    def body(self, cfa_n: torch.Tensor, masks: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
        """Full-resolution correction from the normalised mosaic."""
        x = torch.cat([cfa_n, masks, clip] if self.pack_masks else [cfa_n, clip], dim=1)
        x = self.enc(F.pixel_unshuffle(x, self.factor))
        skips: list[torch.Tensor] = []
        for down, block in zip(self.downs, self.down_blocks):
            skips.append(x)
            x = block(down(x))
        for up, block in zip(self.ups, self.up_blocks):
            x = block(torch.cat([up(x), skips.pop()], dim=1))
        return F.pixel_shuffle(self.head(x), self.factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The mean, the division and the multiplication back stay in float32 whatever the
        # autocast setting: normalised values have no upper bound.
        x = x.float()
        cfa, masks, clip = x[:, 0:1], x[:, 1:4], x[:, 4:5]
        s = cfa.mean(dim=(2, 3), keepdim=True).clamp(min=MEAN_FLOOR)
        if self.training and self.gain_jitter_stops > 0:
            # The network sees patches whose mean is not exactly 1; the same perturbed
            # scale divides and multiplies back, so the output stays in raw units.
            s = s * torch.pow(2.0, (torch.rand_like(s) * 2.0 - 1.0) * self.gain_jitter_stops)
        cfa_n = cfa / s
        delta = self.body(cfa_n, masks, clip).float()
        return (cfa_n * masks + delta) * s


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for period in (6, 2):
        m = XTransUNet(base_width=16, cfa_period=period)
        y = m(torch.rand(1, 5, 288, 288))
        print(f"cfa_period={period}: {count_parameters(m):,} parameters, output {tuple(y.shape)}")
