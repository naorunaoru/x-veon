"""
Gated-convolution U-Net for highlight reconstruction (second pass).

Input:  4 channels (clipped RGB + soft clip mask)
Output: 3 channels (recovered RGB)

Residual design: output = clipped_rgb + model_delta
The model only learns corrections in clipped/near-clipped regions.

Gated convolutions (Yu et al. 2019) learn per-pixel soft gates so the
network can decide which pixels to trust vs. hallucinate, rather than
relying on a hand-crafted mask propagation rule.
"""

import torch
import torch.nn as nn


class GatedConv2d(nn.Module):
    """Gated convolution: features * sigmoid(gates)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.feat_conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                                   stride=stride, padding=padding)
        self.gate_conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                                   stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feat_conv(x) * torch.sigmoid(self.gate_conv(x))


class GatedBlock(nn.Module):
    """Two gated convolutions with ReLU after first."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = GatedConv2d(in_ch, out_ch)
        self.conv2 = GatedConv2d(out_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        return self.conv2(x)


class GatedDown(nn.Module):
    """Downsample with strided gated conv then GatedBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = GatedConv2d(in_ch, out_ch, kernel_size=3,
                                stride=2, padding=1)
        self.block = GatedBlock(out_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.relu(self.down(x)))


class GatedUp(nn.Module):
    """Upsample + skip concat + GatedBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = GatedBlock(out_ch * 2, out_ch)  # *2 for skip concat

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class HighlightUNet(nn.Module):
    """
    Gated-convolution U-Net for highlight reconstruction.

    3 encoder levels (lighter than demosaic model — highlight recovery
    is smoother/lower-frequency than CFA interpolation).

    Channel widths: base_width * [1, 2, 4, 8]
    Default base_width=32 → 32/64/128/256 (~1.2M params)
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 base_width: int = 32):
        super().__init__()
        w = base_width

        # Encoder
        self.enc1 = GatedBlock(in_channels, w)
        self.enc2 = GatedDown(w, w * 2)
        self.enc3 = GatedDown(w * 2, w * 4)

        # Bottleneck
        self.bottleneck = GatedDown(w * 4, w * 8)

        # Decoder
        self.dec3 = GatedUp(w * 8, w * 4)
        self.dec2 = GatedUp(w * 4, w * 2)
        self.dec1 = GatedUp(w * 2, w)

        # Output: 1x1 conv, no activation (residual added outside)
        self.out_conv = nn.Conv2d(w, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3]  # clipped RGB passthrough for residual

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return rgb + self.out_conv(d1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys

    base_width = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    model = HighlightUNet(base_width=base_width)
    print(f"base_width={base_width}, Parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(1, 4, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
