"""
X-Trans CFA pattern definition and utilities.

The X-Trans 6×6 CFA layout (verified from actual RAF files — both X-T10
and X-T2 use this same pattern):

    r b G b r G
    G G r G G b
    G G b G G r
    b r G r b G
    G G b G G r
    G G r G G b

Where R=0, G=1, B=2.

20 of 36 positions are green (55.6%), 8 red (22.2%), 8 blue (22.2%).
"""

import torch
import numpy as np

XTRANS_PATTERN = np.array([
    [0, 2, 1, 2, 0, 1],
    [1, 1, 0, 1, 1, 2],
    [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1],
    [1, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 2],
], dtype=np.int32)


def make_cfa_mask(h: int, w: int) -> torch.Tensor:
    """
    Create the X-Trans CFA pattern tiled to (h, w).

    Returns:
        Tensor of shape (h, w) with values 0 (R), 1 (G), 2 (B).
    """
    tiles_h = (h + 5) // 6
    tiles_w = (w + 5) // 6
    tiled = np.tile(XTRANS_PATTERN, (tiles_h, tiles_w))[:h, :w]
    return torch.from_numpy(tiled.copy()).long()


def make_channel_masks(h: int, w: int) -> torch.Tensor:
    """
    Create binary channel masks for R, G, B positions.

    Returns:
        Tensor of shape (3, h, w) with binary masks for each channel.
    """
    cfa = make_cfa_mask(h, w)
    masks = torch.zeros(3, h, w, dtype=torch.float32)
    masks[0] = (cfa == 0).float()  # R
    masks[1] = (cfa == 1).float()  # G
    masks[2] = (cfa == 2).float()  # B
    return masks


def mosaic(rgb: torch.Tensor, cfa: torch.Tensor) -> torch.Tensor:
    """
    Simulate X-Trans mosaicing: sample RGB image through CFA pattern.

    Args:
        rgb: (3, H, W) or (B, 3, H, W) float tensor
        cfa: (H, W) long tensor with values 0, 1, 2

    Returns:
        (1, H, W) or (B, 1, H, W) single-channel CFA image.
    """
    if rgb.dim() == 3:
        h, w = rgb.shape[1], rgb.shape[2]
        cfa_exp = cfa.unsqueeze(0).expand(1, h, w)
        return torch.gather(rgb, 0, cfa_exp)
    elif rgb.dim() == 4:
        b, _, h, w = rgb.shape
        cfa_exp = cfa.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
        return torch.gather(rgb, 1, cfa_exp)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {rgb.dim()}D")
