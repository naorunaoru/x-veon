"""
U-Net for X-Trans demosaicing.

Architecture: encoder-decoder with skip connections.
- Input: 5 channels (CFA + position masks + clip ratio)
- Output: 3 channels (RGB)
- Additive residual: output = CFA_broadcast + learned_delta
- 4 levels: 64 -> 128 -> 256 -> 512
- 3x3 convolutions throughout
- Receptive field easily covers 2-3 X-Trans repeats (12-18 pixels)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
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


class HighlightHead(nn.Module):
    """Highlight reconstruction head using multi-scale decoder features.

    Instead of dilated convolutions (which cause gridding/ringing artifacts),
    this head fuses features from multiple decoder scales to get large
    receptive field naturally through the U-Net's spatial hierarchy.
    Each scale is projected to a narrow width, upsampled to full resolution,
    and concatenated before fusion convolutions produce the RGB correction.
    """

    def __init__(self, base_width: int = 64, head_width: int | None = None):
        super().__init__()
        w = base_width
        hw = head_width if head_width is not None else base_width
        # 1x1 projections for each decoder scale (+1 on d1 for clip_ratio)
        self.proj1 = nn.Conv2d(w + 1, hw, 1)      # d1 (H) + clip_ratio
        self.proj2 = nn.Conv2d(w * 2, hw, 1)       # d2 (H/2)
        self.proj3 = nn.Conv2d(w * 4, hw, 1)       # d3 (H/4)
        self.proj4 = nn.Conv2d(w * 8, hw, 1)       # d4 (H/8)

        # Fuse all scales at full resolution
        self.fuse = nn.Sequential(
            nn.Conv2d(hw * 4, hw, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hw, hw, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hw, 3, 1),
        )

    def forward(self, d1: torch.Tensor, d2: torch.Tensor,
                d3: torch.Tensor, d4: torch.Tensor,
                clip_ratio: torch.Tensor) -> torch.Tensor:
        h, w = d1.shape[2:]
        f1 = torch.relu(self.proj1(torch.cat([d1, clip_ratio], dim=1)))
        f2 = nn.functional.interpolate(
            torch.relu(self.proj2(d2)), size=(h, w), mode='bilinear', align_corners=False)
        f3 = nn.functional.interpolate(
            torch.relu(self.proj3(d3)), size=(h, w), mode='bilinear', align_corners=False)
        f4 = nn.functional.interpolate(
            torch.relu(self.proj4(d4)), size=(h, w), mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([f1, f2, f3, f4], dim=1))


class XTransUNet(nn.Module):
    """
    U-Net for X-Trans demosaicing.

    4 encoder levels, 4 decoder levels, skip connections at each level.
    Channel widths: base_width * [1, 2, 4, 8, 16] (default 64 → 64..1024).
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 3,
                 base_width: int = 64, hl_head: bool = False):
        super().__init__()
        w = base_width

        # Encoder
        self.enc1 = ConvBlock(in_channels, w)
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

        # Optional highlight reconstruction head
        self.highlight_head = HighlightHead(base_width=w) if hl_head else None

    def forward(self, x):
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

        main_rgb = baseline + self.out_conv(d1)  # 3, H, W

        if self.highlight_head is not None:
            clip_ratio = x[:, 4:5]  # (B, 1, H, W)
            main_rgb = main_rgb + self.highlight_head(d1, d2, d3, d4, clip_ratio)

        return main_rgb


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys

    base_width = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    hl_head = "--hl-head" in sys.argv
    model = XTransUNet(base_width=base_width, hl_head=hl_head)
    print(f"base_width={base_width}, hl_head={hl_head}, Parameters: {count_parameters(model):,}")
    if hl_head:
        hl_params = sum(p.numel() for p in model.highlight_head.parameters())
        print(f"  HighlightHead: {hl_params:,} params")

    # Test forward pass
    x = torch.randn(1, 5, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
