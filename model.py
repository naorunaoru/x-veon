"""
U-Net for X-Trans demosaicing.

Architecture: encoder-decoder with skip connections.
- Input: 4 channels (CFA + position masks)
- Output: 3 channels (RGB)
- 4 levels: 64 -> 128 -> 256 -> 512
- 3x3 convolutions throughout
- Receptive field easily covers 2-3 X-Trans repeats (12-18 pixels)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample with MaxPool then ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample with ConvTranspose2d, concatenate skip, then ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)  # *2 for skip concat

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class XTransUNet(nn.Module):
    """
    U-Net for X-Trans demosaicing.

    4 encoder levels, 4 decoder levels, skip connections at each level.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = DownBlock(512, 1024)

        # Decoder
        self.dec4 = UpBlock(1024, 512)
        self.dec3 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Global residual: CFA broadcast as baseline for all 3 channels.
        # Model learns color deltas, not absolute values → exposure-agnostic.
        cfa = x[:, 0:1]  # (B, 1, H, W)
        baseline = cfa.expand(-1, 3, -1, -1)  # (B, 3, H, W)

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
    model = XTransUNet()
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(1, 4, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
