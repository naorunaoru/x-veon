# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2024-present X-Veon contributors
# Based on darktable's inpaint-opposed and segmentation-based highlight recovery.
"""Highlight reconstruction for post-demosaic RGB data.

Two-pass approach operating on full RGB images (post-WB):
1. Inpaint-opposed: estimates clipped pixels from opposed-channel reference
   averages + global chrominance correction. Fast baseline pass.
2. Segmentation-based: segments clipped regions per color plane, finds best
   unclipped candidate per segment, transfers pseudo-chrominance.

Adapted from the CFA-domain version for use after neural-net demosaic + WB.
Clipping is assumed to be WB-induced (channels pushed past clip_level).
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Constants (from darktable)
# ---------------------------------------------------------------------------

HL_POWERF = 3.0          # cube-root/cube for perceptual linearity
HL_BORDER = 8            # plane border padding
# CLIP_MAGIC = 0.987       # darktable's clip threshold factor
CLIP_MAGIC = 0.9
MIN_SEGMENT_SIZE = 4
SEG_ID_MASK = 0x40000
MAX_SLOTS = SEG_ID_MASK - 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KERNEL_3X3 = np.ones((3, 3), dtype=np.float32)


def _compute_refavg_cr(rgb: np.ndarray) -> np.ndarray:
    """Opposed-channel reference average in cube-root space.

    For each pixel and channel, computes the mean of the *other* two channels'
    3x3 neighborhood averages in cube-root space.

    Args:
        rgb: (H, W, 3) float32

    Returns:
        (H, W, 3) float32, cube-root-space opposed reference per channel
    """
    rgb_pos = np.maximum(rgb, 0).astype(np.float32)

    cr = []
    for c in range(3):
        avg_c = cv2.filter2D(rgb_pos[:, :, c], cv2.CV_32F, _KERNEL_3X3,
                             borderType=cv2.BORDER_REPLICATE) / 9.0
        cr.append(np.cbrt(avg_c))

    refavg = np.empty_like(rgb)
    refavg[:, :, 0] = 0.5 * (cr[1] + cr[2])  # opposed for R
    refavg[:, :, 1] = 0.5 * (cr[0] + cr[2])  # opposed for G
    refavg[:, :, 2] = 0.5 * (cr[0] + cr[1])  # opposed for B
    return refavg


# ---------------------------------------------------------------------------
# Pass 1: Inpaint-opposed
# ---------------------------------------------------------------------------

def reconstruct_opposed(
    rgb: np.ndarray,
    clip_levels: np.ndarray,
    clip_threshold: float = CLIP_MAGIC,
) -> np.ndarray:
    """Inpaint-opposed highlight reconstruction on RGB data.

    For clipped pixels, estimates the true value from the opposed-channel
    reference average (in cube-root space) plus a chrominance correction
    computed from pixels near the clipped areas.

    Args:
        rgb: (H, W, 3) float32, post-WB linear RGB
        clip_levels: (3,) per-channel clip levels (typically [1, 1, 1])
        clip_threshold: fraction of clip_level to treat as clipped

    Returns:
        (H, W, 3) float32, RGB with clipped pixels extended
    """
    h, w, _ = rgb.shape

    clips = clip_levels * clip_threshold
    lo_clips = 0.2 * clips

    clipped = rgb >= clips[np.newaxis, np.newaxis, :]
    if not np.any(clipped):
        return rgb.copy()

    # Clip mask per channel, dilated to find chrominance samples nearby
    max_clip = clips.max()
    dilated_masks = []
    for c in range(3):
        ch_clip = clipped[:, :, c].astype(np.uint8)
        ratio = max_clip / max(clips[c], 1e-6)
        dil_size = int(np.clip(7 * ratio, 7, 21)) | 1  # odd, 7-21
        dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (dil_size, dil_size))
        dilated_masks.append(cv2.dilate(ch_clip, dil_kern))

    # Opposed reference in linear space (cube the cube-root refavg)
    refavg_linear = _compute_refavg_cr(rgb) ** HL_POWERF

    # Chrominance from unclipped pixels within dilated mask
    chrom = np.zeros(3, dtype=np.float64)
    chrom_cnt = np.zeros(3, dtype=np.int64)
    for c in range(3):
        valid = (~clipped[:, :, c]
                 & (rgb[:, :, c] > lo_clips[c])
                 & (dilated_masks[c] > 0))
        # Exclude 3-pixel border
        valid[:3, :] = False; valid[-3:, :] = False
        valid[:, :3] = False; valid[:, -3:] = False

        n = int(np.sum(valid))
        if n > 100:
            diff = (rgb[:, :, c][valid].astype(np.float64)
                    - refavg_linear[:, :, c][valid].astype(np.float64))
            chrom[c] = diff.mean()
            chrom_cnt[c] = n

    # Extend clipped pixels
    out = rgb.copy()
    n_fixed = 0
    for c in range(3):
        ch_clipped = clipped[:, :, c]
        n = int(np.sum(ch_clipped))
        if n == 0:
            continue
        out[:, :, c][ch_clipped] = np.maximum(
            rgb[:, :, c][ch_clipped],
            (refavg_linear[:, :, c] + chrom[c])[ch_clipped],
        )
        n_fixed += n

    if n_fixed > 0:
        print(f"  Highlights (opposed): {n_fixed:,} pixels extended, "
              f"chroma R={chrom[0]:.4f} G={chrom[1]:.4f} B={chrom[2]:.4f} "
              f"(samples R={chrom_cnt[0]} G={chrom_cnt[1]} B={chrom_cnt[2]})")
    return out


# ---------------------------------------------------------------------------
# Pass 2: Segmentation-based reconstruction
# ---------------------------------------------------------------------------

class _Segmentation:
    """Flood-fill segmentation data structure."""
    __slots__ = ('data', 'tmp', 'size', 'xmin', 'xmax', 'ymin', 'ymax',
                 'val1', 'val2', 'nr', 'border', 'slots', 'width', 'height')

    def __init__(self, width: int, height: int, border: int, max_slots: int):
        slots = max(256, min(max_slots, MAX_SLOTS))
        n = width * height
        self.data = np.zeros(n, dtype=np.uint32)
        self.tmp = np.zeros(n, dtype=np.uint32)
        self.size = np.zeros(slots, dtype=np.int32)
        self.xmin = np.zeros(slots, dtype=np.int32)
        self.xmax = np.zeros(slots, dtype=np.int32)
        self.ymin = np.zeros(slots, dtype=np.int32)
        self.ymax = np.zeros(slots, dtype=np.int32)
        self.val1 = np.zeros(slots, dtype=np.float32)  # candidate value
        self.val2 = np.zeros(slots, dtype=np.float32)  # candidate refavg
        self.nr = 2
        self.border = border
        self.slots = slots
        self.width = width
        self.height = height


def _clear_slot(seg: _Segmentation, sid: int):
    seg.size[sid] = 0
    seg.xmin[sid] = 0; seg.xmax[sid] = 0
    seg.ymin[sid] = 0; seg.ymax[sid] = 0
    seg.val1[sid] = 0; seg.val2[sid] = 0


def _get_seg_id(seg: _Segmentation, loc: int) -> int:
    if loc >= seg.width * (seg.height - seg.border):
        return 0
    sid = int(seg.data[loc]) & (SEG_ID_MASK - 1)
    return sid if (1 < sid < seg.nr) else 0


def _floodfill(yin: int, xin: int, seg: _Segmentation, sid: int) -> bool:
    """Flood-fill segmentation from seed pixel."""
    if sid >= seg.slots - 2:
        return False

    w, h, border = seg.width, seg.height, seg.border
    d = seg.data
    stack = [(xin, yin)]
    min_x, max_x, min_y, max_y = xin, xin, yin, yin
    cnt = 0
    _clear_slot(seg, sid)

    while stack:
        x, y = stack.pop()
        if d[y * w + x] != 1:
            continue

        d[y * w + x] = sid
        cnt += 1
        min_x = min(min_x, x); max_x = max(max_x, x)
        min_y = min(min_y, y); max_y = max(max_y, y)

        y_up, y_down = y - 1, y + 1

        # Scan right
        xr = x + 1
        while xr < w - border and d[y * w + xr] == 1:
            d[y * w + xr] = sid
            cnt += 1
            max_x = max(max_x, xr)
            if y_up >= border and d[y_up * w + xr] == 1:
                stack.append((xr, y_up))
            elif y_up >= border and d[y_up * w + xr] == 0:
                d[y_up * w + xr] = SEG_ID_MASK | sid
            if y_down < h - border and d[y_down * w + xr] == 1:
                stack.append((xr, y_down))
            elif y_down < h - border and d[y_down * w + xr] == 0:
                d[y_down * w + xr] = SEG_ID_MASK | sid
            xr += 1

        # Scan left
        xl = x - 1
        while xl >= border and d[y * w + xl] == 1:
            d[y * w + xl] = sid
            cnt += 1
            min_x = min(min_x, xl)
            if y_up >= border and d[y_up * w + xl] == 1:
                stack.append((xl, y_up))
            elif y_up >= border and d[y_up * w + xl] == 0:
                d[y_up * w + xl] = SEG_ID_MASK | sid
            if y_down < h - border and d[y_down * w + xl] == 1:
                stack.append((xl, y_down))
            elif y_down < h - border and d[y_down * w + xl] == 0:
                d[y_down * w + xl] = SEG_ID_MASK | sid
            xl -= 1

        # Check above/below at seed x
        if y_up >= border and d[y_up * w + x] == 1:
            stack.append((x, y_up))
        elif y_up >= border and d[y_up * w + x] == 0:
            d[y_up * w + x] = SEG_ID_MASK | sid
        if y_down < h - border and d[y_down * w + x] == 1:
            stack.append((x, y_down))
        elif y_down < h - border and d[y_down * w + x] == 0:
            d[y_down * w + x] = SEG_ID_MASK | sid

    if cnt < MIN_SEGMENT_SIZE:
        # Too small — revert
        for row in range(min_y, max_y + 1):
            for col in range(min_x, max_x + 1):
                loc = row * w + col
                if d[loc] == sid:
                    d[loc] = 1
                elif d[loc] == (sid | SEG_ID_MASK):
                    d[loc] = 0
        return False

    seg.size[sid] = cnt
    seg.xmin[sid] = min_x; seg.xmax[sid] = max_x
    seg.ymin[sid] = min_y; seg.ymax[sid] = max_y
    seg.nr += 1
    _clear_slot(seg, sid + 1)
    return True


def _segmentize_plane(seg: _Segmentation):
    """Scan and flood-fill all marked pixels."""
    w, h, border = seg.width, seg.height, seg.border
    sid = 2
    for row in range(border, h - border):
        for col in range(border, w - border):
            if sid >= seg.slots - 2:
                return
            if seg.data[row * w + col] == 1:
                if _floodfill(row, col, seg, sid):
                    sid += 1


def _morphological_close(seg: _Segmentation, radius: int):
    """Morphological closing (dilate then erode) using cv2."""
    if radius <= 0:
        return

    w, h, border = seg.width, seg.height, seg.border
    d = seg.data.reshape(h, w)

    # Zero borders
    d[:border, :] = 0
    d[h - border:, :] = 0
    d[:, :border] = 0
    d[:, w - border:] = 0

    binary = (d > 0).astype(np.uint8)

    # Dilate with radius, erode with max(radius-1, 1) — matches original
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    erode_r = max(radius - 1, 1)
    ero_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erode_r + 1, 2 * erode_r + 1))

    closed = cv2.erode(cv2.dilate(binary, dil_kernel), ero_kernel)

    d[:] = closed.astype(np.uint32)

    # Re-zero borders
    d[:border, :] = 0
    d[h - border:, :] = 0
    d[:, :border] = 0
    d[:, w - border:] = 0


def _local_std_dev(plane: np.ndarray, idx: int, w: int) -> float:
    """5x5 cross-shaped local standard deviation."""
    offsets = [
        -2*w - 1, -2*w, -2*w + 1,
        -w - 2, -w - 1, -w, -w + 1, -w + 2,
        -2, -1, 0, 1, 2,
        w - 2, w - 1, w, w + 1, w + 2,
        2*w - 1, 2*w, 2*w + 1,
    ]
    vals = [float(plane[idx + o]) for o in offsets]
    n = len(vals)
    avg = sum(vals) / n
    var = sum((v - avg) ** 2 for v in vals) / n
    return var ** 0.5


def _calc_weight(plane: np.ndarray, loc: int, w: int, clipval: float) -> float:
    """Weight for candidate selection: smoothness * brightness."""
    smoothness = max(0.0, 1.0 - 10.0 * _local_std_dev(plane, loc, w) ** 0.5)
    # 3x3 mean
    val = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            val += float(plane[loc + dy * w + dx])
    val /= 9.0
    sval = max(1.0, (min(clipval, val) / clipval) ** 2)
    return sval * smoothness


# Gaussian 5x5 weights for candidate averaging
_GAUSS_5X5 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
], dtype=np.float32)


def _calc_plane_candidates(
    plane: np.ndarray, refavg: np.ndarray, seg: _Segmentation,
    clipval: float, badlevel: float,
):
    """Find best unclipped candidate per segment."""
    w = seg.width
    for sid in range(2, seg.nr):
        seg.val1[sid] = 0
        seg.val2[sid] = 0

        if (seg.ymax[sid] - seg.ymin[sid] <= 2) or (seg.xmax[sid] - seg.xmin[sid] <= 2):
            continue

        test_ref = -1
        test_weight = 0.0

        row_min = max(seg.border + 2, seg.ymin[sid] - 2)
        row_max = min(seg.height - seg.border - 2, seg.ymax[sid] + 3)
        col_min = max(seg.border + 2, seg.xmin[sid] - 2)
        col_max = min(seg.width - seg.border - 2, seg.xmax[sid] + 3)

        for row in range(row_min, row_max):
            for col in range(col_min, col_max):
                pos = row * w + col
                pid = _get_seg_id(seg, pos)
                if pid == sid and plane[pos] < clipval:
                    is_border = 1.0 if (seg.data[pos] & SEG_ID_MASK) else 0.75
                    wht = _calc_weight(plane, pos, w, clipval) * is_border
                    if wht > test_weight:
                        test_weight = wht
                        test_ref = pos

        if test_ref >= 0 and test_weight > 1.0 - badlevel:
            total = 0.0
            pix = 0.0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    pos = test_ref + dy * w + dx
                    gw = _GAUSS_5X5[dy + 2, dx + 2]
                    if plane[pos] < clipval:
                        total += float(plane[pos]) * gw
                        pix += gw
            avg = total / max(1.0, pix)
            if avg > 0.125 * clipval:
                seg.val1[sid] = min(clipval, avg)
                seg.val2[sid] = refavg[test_ref]


def _extend_border(arr: np.ndarray, width: int, height: int, border: int):
    """Extend plane data to fill border region."""
    if border <= 0:
        return
    a = arr.reshape(height, width)

    # Left/right: replicate border columns
    a[border:height - border, :border] = a[border:height - border, border:border + 1]
    a[border:height - border, width - border:] = a[border:height - border, width - border - 1:width - border]

    # Top/bottom: replicate border rows with column clamping
    cols = np.arange(width)
    clamped = np.clip(cols, border, width - border - 1)
    top_row = a[border, clamped]
    bot_row = a[height - border - 1, clamped]
    a[:border, :] = top_row[np.newaxis, :]
    a[height - border:, :] = bot_row[np.newaxis, :]


def reconstruct_segmented(
    rgb: np.ndarray,
    clip_levels: np.ndarray,
    clip_threshold: float = CLIP_MAGIC,
    combine_radius: int = 2,
    candidating: float = 0.5,
    original_rgb: np.ndarray = None,
) -> np.ndarray:
    """Segmentation-based highlight reconstruction on RGB data.

    Downsamples each channel by 3x, operates in cube-root space.
    Uses flood-fill segmentation, morphological closing, and candidate
    selection to reconstruct clipped pixels.

    Args:
        rgb: (H, W, 3) float32, post-WB RGB (may be pre-processed by opposed)
        clip_levels: (3,) per-channel clip levels
        clip_threshold: fraction of clip_level to treat as clipped
        combine_radius: morphological closing radius (0-5)
        candidating: candidate acceptance threshold (0-1, lower = stricter)
        original_rgb: if provided, used for per-pixel refavg

    Returns:
        (H, W, 3) float32, RGB with reconstructed highlights
    """
    h, w, _ = rgb.shape
    original = original_rgb if original_rgb is not None else rgb

    clips = clip_levels * clip_threshold
    cube_clips = [np.cbrt(float(clips[c])) for c in range(3)]

    # Plane dimensions (1/3 resolution + border)
    round_even = lambda n: (n + 1) & ~1
    pwidth = round_even(w // 3) + 2 * HL_BORDER
    pheight = round_even(h // 3) + 2 * HL_BORDER
    psize = pwidth * pheight

    n_sp_rows = h // 3
    n_sp_cols = w // 3

    # Allocate planes and segmentations
    planes = [np.zeros(psize, dtype=np.float32) for _ in range(3)]
    refavgs = [np.zeros(psize, dtype=np.float32) for _ in range(3)]

    max_segments = max(256, (w * h) // 4000)
    segs = [_Segmentation(pwidth, pheight, HL_BORDER + 1, max_segments)
            for _ in range(3)]

    # Step 1: Build downsampled color planes via box-filter (3x3 area resize)
    rgb_f = rgb.astype(np.float32)
    channel_ds_cbrt = []
    for c in range(3):
        ds = cv2.resize(rgb_f[:, :, c], (n_sp_cols, n_sp_rows),
                        interpolation=cv2.INTER_AREA)
        ds_cbrt = np.cbrt(np.maximum(ds, 0))
        channel_ds_cbrt.append(ds_cbrt)

    # Compute refavg at downsampled resolution: opposed channels
    refavg_ds = [
        0.5 * (channel_ds_cbrt[1] + channel_ds_cbrt[2]),  # for R
        0.5 * (channel_ds_cbrt[0] + channel_ds_cbrt[2]),  # for G
        0.5 * (channel_ds_cbrt[0] + channel_ds_cbrt[1]),  # for B
    ]

    any_clipped = 0
    for c in range(3):
        plane_2d = planes[c].reshape(pheight, pwidth)
        plane_2d[HL_BORDER:HL_BORDER + n_sp_rows,
                 HL_BORDER:HL_BORDER + n_sp_cols] = channel_ds_cbrt[c]

        ref_2d = refavgs[c].reshape(pheight, pwidth)
        ref_2d[HL_BORDER:HL_BORDER + n_sp_rows,
               HL_BORDER:HL_BORDER + n_sp_cols] = refavg_ds[c]

        # Mark clipped superpixels
        clip_mask = channel_ds_cbrt[c] >= cube_clips[c]
        seg_2d = segs[c].data.reshape(pheight, pwidth)
        seg_2d[HL_BORDER:HL_BORDER + n_sp_rows,
               HL_BORDER:HL_BORDER + n_sp_cols] |= clip_mask.astype(np.uint32)
        any_clipped += int(np.sum(clip_mask))

    if any_clipped < 20:
        return rgb.copy()

    # Step 2: Extend border data
    for c in range(3):
        _extend_border(planes[c], pwidth, pheight, HL_BORDER)

    # Step 3: Morphological closing + segmentation
    for c in range(3):
        _morphological_close(segs[c], combine_radius)
        _segmentize_plane(segs[c])

    # Step 4: Find best candidates per segment
    for c in range(3):
        _calc_plane_candidates(planes[c], refavgs[c], segs[c],
                               cube_clips[c], candidating)

    # Step 5: Reconstruct clipped pixels
    out = rgb.copy()
    original_f = np.maximum(original.astype(np.float32), 0)
    clipped = original_f >= clips[np.newaxis, np.newaxis, :]

    # Full-resolution refavg for original image
    refavg_cr_map = _compute_refavg_cr(original)

    n_fixed = 0
    for c in range(3):
        ch_clipped = clipped[:, :, c].copy()
        # Exclude 1-pixel border
        ch_clipped[0, :] = False; ch_clipped[-1, :] = False
        ch_clipped[:, 0] = False; ch_clipped[:, -1] = False

        if not np.any(ch_clipped):
            continue

        rows, cols = np.where(ch_clipped)

        # Map to plane coordinates
        p_rows = HL_BORDER + rows // 3
        p_cols = HL_BORDER + cols // 3

        # Clip to plane bounds
        valid_plane = ((p_rows >= 0) & (p_rows < pheight) &
                       (p_cols >= 0) & (p_cols < pwidth))
        rows = rows[valid_plane]
        cols = cols[valid_plane]
        p_rows = p_rows[valid_plane]
        p_cols = p_cols[valid_plane]

        seg = segs[c]
        seg_2d = seg.data.reshape(pheight, pwidth)

        # Get segment IDs
        sids = seg_2d[p_rows, p_cols].astype(np.int64) & (SEG_ID_MASK - 1)
        plane_locs = p_rows * pwidth + p_cols
        in_bounds = plane_locs < (pwidth * (pheight - seg.border))
        valid_sid = in_bounds & (sids > 1) & (sids < seg.nr)

        if not np.any(valid_sid):
            continue

        # Index candidates
        sids_safe = np.clip(sids, 0, seg.slots - 1)
        candidates = seg.val1[sids_safe]
        cand_refs = seg.val2[sids_safe]

        has_candidate = valid_sid & (candidates != 0)
        if not np.any(has_candidate):
            continue

        hc_rows = rows[has_candidate]
        hc_cols = cols[has_candidate]
        hc_cand = candidates[has_candidate]
        hc_cref = cand_refs[has_candidate]

        refavg_here = refavg_cr_map[hc_rows, hc_cols, c]
        oval = (refavg_here + hc_cand - hc_cref) ** HL_POWERF
        invals = original_f[hc_rows, hc_cols, c]
        out[hc_rows, hc_cols, c] = np.maximum(invals, oval)
        n_fixed += len(hc_rows)

    if n_fixed > 0:
        print(f"  Highlights (segmented): {n_fixed:,} pixels reconstructed")
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def reconstruct_highlights(
    rgb: np.ndarray,
    clip_levels: np.ndarray,
    clip_threshold: float = CLIP_MAGIC,
    combine_radius: int = 2,
    candidating: float = 0.5,
) -> np.ndarray:
    """Two-pass highlight reconstruction on post-demosaic RGB data.

    Pass 1: inpaint-opposed (fast baseline)
    Pass 2: segmentation-based (structured recovery)

    Args:
        rgb: (H, W, 3) float32, post-WB linear RGB
        clip_levels: (3,) per-channel clip levels (typically [1.0, 1.0, 1.0])
        clip_threshold: fraction of clip_level to treat as clipped
        combine_radius: morphological closing radius for segmentation
        candidating: candidate acceptance threshold for segmentation

    Returns:
        (H, W, 3) float32, RGB with reconstructed highlights
    """
    original = rgb.copy()

    # Pass 1: opposed inpainting
    result = reconstruct_opposed(rgb, clip_levels, clip_threshold)

    # Pass 2: segmentation-based refinement (using original for refavg)
    result = reconstruct_segmented(
        result, clip_levels, clip_threshold,
        combine_radius, candidating, original_rgb=original,
    )

    return result
