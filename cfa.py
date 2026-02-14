"""
CFA (Color Filter Array) pattern definitions and utilities.

Supports:
- X-Trans 6x6 CFA (Fuji ILC sensors)
- Bayer 2x2 CFA (canonical RGGB; other variants are phase-shifted)

Color encoding: R=0, G=1, B=2 throughout.
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Pattern constants
# ---------------------------------------------------------------------------

XTRANS_PATTERN = np.array([
    [0, 2, 1, 2, 0, 1],
    [1, 1, 0, 1, 1, 2],
    [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1],
    [1, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 2],
], dtype=np.int32)

# Canonical Bayer pattern (RGGB). BGGR/GRBG/GBRG are just phase shifts of
# this same pattern — handled at inference time via find_pattern_shift().
BAYER_PATTERN = np.array([
    [0, 1],
    [1, 2],
], dtype=np.int32)

CFA_REGISTRY: dict[str, np.ndarray] = {
    "xtrans": XTRANS_PATTERN,
    "bayer": BAYER_PATTERN,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cfa_period(pattern: np.ndarray) -> int:
    """Return the spatial period of a CFA pattern (6 for X-Trans, 2 for Bayer)."""
    return pattern.shape[0]


def patch_alignment(pattern: np.ndarray) -> int:
    """Required patch size alignment: lcm(CFA period, UNet factor 16)."""
    return math.lcm(cfa_period(pattern), 16)


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def make_cfa_mask(h: int, w: int, pattern: np.ndarray = XTRANS_PATTERN) -> torch.Tensor:
    """
    Tile a CFA pattern to (h, w).

    Returns:
        Tensor of shape (h, w) with values 0 (R), 1 (G), 2 (B).
    """
    ph, pw = pattern.shape
    tiles_h = (h + ph - 1) // ph
    tiles_w = (w + pw - 1) // pw
    tiled = np.tile(pattern, (tiles_h, tiles_w))[:h, :w]
    return torch.from_numpy(tiled.copy()).long()


def make_channel_masks(h: int, w: int, pattern: np.ndarray = XTRANS_PATTERN) -> torch.Tensor:
    """
    Binary channel masks for R, G, B positions.

    Returns:
        Tensor of shape (3, h, w).
    """
    cfa = make_cfa_mask(h, w, pattern)
    masks = torch.zeros(3, h, w, dtype=torch.float32)
    masks[0] = (cfa == 0).float()  # R
    masks[1] = (cfa == 1).float()  # G
    masks[2] = (cfa == 2).float()  # B
    return masks


def mosaic(rgb: torch.Tensor, cfa: torch.Tensor) -> torch.Tensor:
    """
    Simulate CFA mosaicing: sample RGB image through CFA pattern.

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


# ---------------------------------------------------------------------------
# Pattern detection (for inference on raw files)
# ---------------------------------------------------------------------------

def find_pattern_shift(raw_pattern: np.ndarray, reference: np.ndarray) -> tuple[int, int]:
    """
    Find (dy, dx) shift of raw_pattern relative to a reference pattern.

    Works for any CFA period (6 for X-Trans, 2 for Bayer).
    """
    period = reference.shape[0]
    tile = raw_pattern[:period, :period].copy()
    tile[tile == 3] = 1  # G2 → G
    for dy in range(period):
        for dx in range(period):
            shifted = np.roll(np.roll(reference, dy, axis=0), dx, axis=1)
            if np.array_equal(tile, shifted):
                return dy, dx
    raise ValueError(f"Could not match CFA pattern to reference (period={period})")


def detect_cfa_from_raw(raw_pattern: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Auto-detect CFA type from rawpy's raw_pattern or raw_colors_visible.

    Returns:
        (cfa_name, canonical_pattern) where cfa_name is a key in CFA_REGISTRY.
    """
    h, w = raw_pattern.shape[:2]

    # libraw's raw_pattern uses 4 color indices: R=0, G1=1, B=2, G2=3.
    # Remap G2 (3) → G (1) so we only deal with {0, 1, 2}.
    if np.any(raw_pattern[:max(h, 6), :max(w, 6)] == 3):
        raw_pattern = raw_pattern.copy()
        raw_pattern[raw_pattern == 3] = 1

    # Try X-Trans (6x6)
    if h >= 6 and w >= 6:
        tile = raw_pattern[:6, :6]
        for dy in range(6):
            for dx in range(6):
                shifted = np.roll(np.roll(XTRANS_PATTERN, dy, axis=0), dx, axis=1)
                if np.array_equal(tile, shifted):
                    return "xtrans", XTRANS_PATTERN

    # Try Bayer (2x2) — any phase of RGGB
    if h >= 2 and w >= 2:
        tile = raw_pattern[:2, :2]
        for dy in range(2):
            for dx in range(2):
                shifted = np.roll(np.roll(BAYER_PATTERN, dy, axis=0), dx, axis=1)
                if np.array_equal(tile, shifted):
                    return "bayer", BAYER_PATTERN

    raise ValueError(
        f"Could not identify CFA pattern from raw_pattern "
        f"(shape={raw_pattern.shape}, top-left 2x2={raw_pattern[:2, :2].tolist()})"
    )
