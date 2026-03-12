# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2024-present X-Veon contributors
# Based on darktable's inpaint-opposed and segmentation-based highlight recovery.
"""Highlight reconstruction for CFA raw data.

Two-pass approach operating on CFA data before demosaicing:
1. Inpaint-opposed: estimates clipped pixels from opposed-channel reference
   averages + global chrominance correction. Fast baseline pass.
2. Segmentation-based: segments clipped regions per color plane, finds best
   unclipped candidate per segment, transfers pseudo-chrominance. More
   sophisticated, handles structured highlights.

Works for both Bayer and X-Trans CFA patterns.
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Constants (from darktable)
# ---------------------------------------------------------------------------

HL_POWERF = 3.0          # cube-root/cube for perceptual linearity
HL_BORDER = 8            # plane border padding
CLIP_MAGIC = 0.987       # darktable's clip threshold factor
MIN_SEGMENT_SIZE = 4
SEG_ID_MASK = 0x40000
MAX_SLOTS = SEG_ID_MASK - 2


# ---------------------------------------------------------------------------
# Vectorized helpers
# ---------------------------------------------------------------------------

def _make_full_pattern(raw_pattern: np.ndarray, h: int, w: int) -> np.ndarray:
    """Tile CFA pattern to full image size, mapping G2 -> G."""
    pat = raw_pattern.copy()
    pat[pat == 3] = 1
    ph, pw = pat.shape
    return np.tile(pat, ((h + ph - 1) // ph, (w + pw - 1) // pw))[:h, :w]


_KERNEL_3X3 = np.ones((3, 3), dtype=np.float32)


def _compute_refavg_cr(cfa: np.ndarray, full_pat: np.ndarray) -> np.ndarray:
    """Vectorized opposed-channel reference average in cube-root space.

    For each pixel, computes the mean of each color channel in the 3x3
    neighborhood, takes the cube root, then averages the two opposed channels.
    Returns the cube-root-space result (not cubed).
    """
    cfa_pos = np.maximum(cfa, 0).astype(np.float32)

    # Per-channel 3x3 sums and counts via convolution
    cr = []
    for c in range(3):
        mask_c = (full_pat == c).astype(np.float32)
        sum_c = cv2.filter2D(cfa_pos * mask_c, cv2.CV_32F, _KERNEL_3X3,
                             borderType=cv2.BORDER_REPLICATE)
        cnt_c = cv2.filter2D(mask_c, cv2.CV_32F, _KERNEL_3X3,
                             borderType=cv2.BORDER_REPLICATE)
        cr.append(np.cbrt(np.divide(sum_c, cnt_c, out=np.zeros_like(sum_c),
                                    where=cnt_c > 0)))

    # Opposed average: for each pixel's channel, average the other two
    opp_r = 0.5 * (cr[1] + cr[2])  # for red pixels
    opp_g = 0.5 * (cr[0] + cr[2])  # for green pixels
    opp_b = 0.5 * (cr[0] + cr[1])  # for blue pixels
    return np.where(full_pat == 0, opp_r,
                    np.where(full_pat == 1, opp_g, opp_b))


# ---------------------------------------------------------------------------
# Pass 1: Inpaint-opposed (vectorized)
# ---------------------------------------------------------------------------

def reconstruct_opposed(
    cfa: np.ndarray,
    raw_pattern: np.ndarray,
    clip_levels: np.ndarray,
    clip_threshold: float = CLIP_MAGIC,
) -> np.ndarray:
    """Inpaint-opposed highlight reconstruction on CFA data.

    For clipped pixels, estimates the true value from the opposed-channel
    reference average (in cube-root space) plus a chrominance correction
    computed from pixels near the clipped areas (dilated mask approach,
    matching darktable).

    Args:
        cfa: (H, W) float32, CFA data (WB-applied or not)
        raw_pattern: (H, W) int, channel index per pixel
        clip_levels: (3,) per-channel clip levels
        clip_threshold: fraction of clip_level to treat as clipped

    Returns:
        (H, W) float32, CFA with clipped pixels extended
    """
    h, w = cfa.shape
    full_pat = _make_full_pattern(raw_pattern, h, w)

    clips = clip_levels * clip_threshold
    lo_clips = 0.2 * clips

    # Per-pixel clip levels
    clip_pp = clips[full_pat]
    lo_clip_pp = lo_clips[full_pat]

    clipped = cfa >= clip_pp
    if not np.any(clipped):
        return cfa.copy()

    # Step 1: Build per-channel clip mask at 1/3 resolution
    mheight, mwidth = h // 3, w // 3
    mask = np.zeros((3, mheight, mwidth), dtype=np.uint8)
    for c in range(3):
        ch_clipped = ((full_pat == c) & clipped).astype(np.float32)
        summed = cv2.filter2D(ch_clipped, cv2.CV_32F, _KERNEL_3X3,
                              borderType=cv2.BORDER_CONSTANT)
        mask[c] = (summed[::3, ::3][:mheight, :mwidth] > 0).astype(np.uint8)
    # Zero borders (original starts at mrow=1, mcol=1)
    mask[:, 0, :] = 0; mask[:, -1, :] = 0
    mask[:, :, 0] = 0; mask[:, :, -1] = 0

    if not np.any(mask):
        return cfa.copy()

    # Step 2: Dilate mask — adaptive per channel.
    # Channels that clip first (lower clip_level) need wider dilation to find
    # enough unclipped chrominance samples nearby.
    max_clip = clips.max()
    dilated = np.zeros_like(mask)
    for c in range(3):
        ratio = max_clip / max(clips[c], 1e-6)
        dil_size = int(np.clip(7 * ratio, 7, 21)) | 1  # odd, 7-21
        dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (dil_size, dil_size))
        dilated[c] = cv2.dilate(mask[c], dil_kern)

    # Step 3: Chrominance from unclipped pixels within dilated mask
    refavg_linear = _compute_refavg_cr(cfa, full_pat) ** HL_POWERF
    not_clipped = ~clipped
    not_too_dim = cfa > lo_clip_pp

    chrom = np.zeros(3, dtype=np.float64)
    chrom_cnt = np.zeros(3, dtype=np.int64)
    for c in range(3):
        # Upsample dilated mask to full resolution via index mapping
        my = np.minimum(np.arange(h) // 3, mheight - 1)
        mx = np.minimum(np.arange(w) // 3, mwidth - 1)
        dilated_up = dilated[c][my[:, None], mx[None, :]]

        valid = ((full_pat == c) & not_clipped & not_too_dim
                 & (dilated_up > 0))
        # Exclude 3-pixel border (matching original)
        valid[:3, :] = False; valid[-3:, :] = False
        valid[:, :3] = False; valid[:, -3:] = False

        n = int(np.sum(valid))
        if n > 100:
            diff = cfa[valid].astype(np.float64) - refavg_linear[valid].astype(np.float64)
            chrom[c] = diff.mean()
            chrom_cnt[c] = n

    # Step 4: Extend clipped pixels
    out = cfa.copy()
    n_fixed = 0
    for c in range(3):
        ch_clipped = (full_pat == c) & clipped
        n = int(np.sum(ch_clipped))
        if n == 0:
            continue
        out[ch_clipped] = np.maximum(
            cfa[ch_clipped],
            (refavg_linear + chrom[c])[ch_clipped],
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
    """Flood-fill segmentation from seed pixel. Faithful port from darktable."""
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
    """Extend plane data to fill border region (vectorized)."""
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
    cfa: np.ndarray,
    raw_pattern: np.ndarray,
    clip_levels: np.ndarray,
    clip_threshold: float = CLIP_MAGIC,
    combine_radius: int = 2,
    candidating: float = 0.5,
    original_cfa: np.ndarray = None,
) -> np.ndarray:
    """Segmentation-based highlight reconstruction on CFA data.

    Operates on 3x3 superpixel color planes in cube-root space.
    Uses flood-fill segmentation, morphological closing, and candidate
    selection to reconstruct clipped pixels.

    Args:
        cfa: (H, W) float32, CFA data (may be pre-processed by opposed pass)
        raw_pattern: (H, W) int, channel index per pixel
        clip_levels: (3,) per-channel clip levels
        clip_threshold: fraction of clip_level to treat as clipped
        combine_radius: morphological closing radius (0-5)
        candidating: candidate acceptance threshold (0-1, lower = stricter)
        original_cfa: if provided, used for per-pixel refavg (darktable uses
            original raw, not opposed-inpainted version)

    Returns:
        (H, W) float32, CFA with reconstructed highlights
    """
    h, w = cfa.shape
    original = original_cfa if original_cfa is not None else cfa
    full_pat = _make_full_pattern(raw_pattern, h, w)

    clips = clip_levels * clip_threshold

    # Determine superpixel alignment
    pat = raw_pattern.copy()
    pat[pat == 3] = 1
    period_h, period_w = pat.shape
    is_bayer = (period_h <= 2 and period_w <= 2)
    is_green_00 = is_bayer and (int(pat[0, 0]) == 1)
    xshifter = 1 if is_green_00 else 2

    # Plane dimensions (1/3 resolution + border)
    round_even = lambda n: (n + 1) & ~1
    pwidth = round_even(w // 3) + 2 * HL_BORDER
    pheight = round_even(h // 3) + 2 * HL_BORDER
    psize = pwidth * pheight

    # Allocate planes and segmentations
    planes = [np.zeros(psize, dtype=np.float32) for _ in range(3)]
    refavgs = [np.zeros(psize, dtype=np.float32) for _ in range(3)]

    cube_clips = [np.cbrt(float(clips[c])) for c in range(3)]

    max_segments = max(256, (w * h) // 4000)
    segs = [_Segmentation(pwidth, pheight, HL_BORDER + 1, max_segments) for _ in range(3)]

    # Step 1: Build downsampled color planes from 3x3 superpixels (vectorized)
    cfa_f = cfa.astype(np.float32)
    channel_cbrt = []
    for c in range(3):
        mask_c = (full_pat == c).astype(np.float32)
        sum_c = cv2.filter2D(cfa_f * mask_c, cv2.CV_32F, _KERNEL_3X3,
                             borderType=cv2.BORDER_REPLICATE)
        cnt_c = cv2.filter2D(mask_c, cv2.CV_32F, _KERNEL_3X3,
                             borderType=cv2.BORDER_REPLICATE)
        mean_c = np.divide(sum_c, cnt_c, out=np.zeros_like(sum_c),
                           where=cnt_c > 0)
        channel_cbrt.append(np.cbrt(mean_c).astype(np.float32))

    # Sample at superpixel centers: row%3==1, col%3==xshifter
    n_sp_rows = len(range(1, h - 1, 3))
    n_sp_cols = len(range(xshifter, w - 1, 3))

    # Compute refavg at superpixel centers
    m0 = channel_cbrt[0][1::3, xshifter::3][:n_sp_rows, :n_sp_cols]
    m1 = channel_cbrt[1][1::3, xshifter::3][:n_sp_rows, :n_sp_cols]
    m2 = channel_cbrt[2][1::3, xshifter::3][:n_sp_rows, :n_sp_cols]

    refavg_sp = [
        0.5 * (m1 + m2),  # for R
        0.5 * (m0 + m2),  # for G
        0.5 * (m0 + m1),  # for B
    ]

    any_clipped = 0
    for c in range(3):
        centers = channel_cbrt[c][1::3, xshifter::3][:n_sp_rows, :n_sp_cols]
        plane_2d = planes[c].reshape(pheight, pwidth)
        plane_2d[HL_BORDER:HL_BORDER + n_sp_rows,
                 HL_BORDER:HL_BORDER + n_sp_cols] = centers

        ref_2d = refavgs[c].reshape(pheight, pwidth)
        ref_2d[HL_BORDER:HL_BORDER + n_sp_rows,
               HL_BORDER:HL_BORDER + n_sp_cols] = refavg_sp[c]

        # Mark clipped superpixels
        clip_mask = centers >= cube_clips[c]
        seg_2d = segs[c].data.reshape(pheight, pwidth)
        seg_2d[HL_BORDER:HL_BORDER + n_sp_rows,
               HL_BORDER:HL_BORDER + n_sp_cols] |= clip_mask.astype(np.uint32)
        any_clipped += int(np.sum(clip_mask))

    if any_clipped < 20:
        return cfa.copy()

    # Step 2: Extend border data (vectorized)
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

    # Step 5: Reconstruct clipped raw pixels (vectorized)
    out = cfa.copy()
    clip_pp = clips[full_pat]
    original_f = np.maximum(original.astype(np.float32), 0)
    clipped = original_f >= clip_pp

    # Precompute cube-root refavg map for original image
    refavg_cr_map = _compute_refavg_cr(original, full_pat)

    n_fixed = 0
    for c in range(3):
        ch_clipped = (full_pat == c) & clipped
        # Exclude 1-pixel border
        ch_clipped[0, :] = False; ch_clipped[-1, :] = False
        ch_clipped[:, 0] = False; ch_clipped[:, -1] = False

        if not np.any(ch_clipped):
            continue

        rows, cols = np.where(ch_clipped)

        # Map to plane coordinates
        p_rows = HL_BORDER + rows // 3
        p_cols = cols // 3 + HL_BORDER

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
        # Bounds check for border
        plane_locs = p_rows * pwidth + p_cols
        in_bounds = plane_locs < (pwidth * (pheight - seg.border))
        valid_sid = in_bounds & (sids > 1) & (sids < seg.nr)

        if not np.any(valid_sid):
            continue

        # Index candidates (clip sids to valid range for safe indexing)
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

        refavg_here = refavg_cr_map[hc_rows, hc_cols]
        oval = (refavg_here + hc_cand - hc_cref) ** HL_POWERF
        invals = original_f[hc_rows, hc_cols]
        out[hc_rows, hc_cols] = np.maximum(invals, oval)
        n_fixed += len(hc_rows)

    if n_fixed > 0:
        print(f"  Highlights (segmented): {n_fixed:,} pixels reconstructed")
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def reconstruct_highlights(
    cfa: np.ndarray,
    raw_pattern: np.ndarray,
    clip_levels: np.ndarray,
    clip_threshold: float = CLIP_MAGIC,
    combine_radius: int = 2,
    candidating: float = 0.5,
) -> np.ndarray:
    """Two-pass highlight reconstruction on CFA data.

    Pass 1: inpaint-opposed (fast baseline)
    Pass 2: segmentation-based (structured recovery)

    Args:
        cfa: (H, W) float32, CFA data
        raw_pattern: (H, W) int, channel index per pixel (0=R, 1=G, 2=B, 3=G2)
        clip_levels: (3,) per-channel clip levels
        clip_threshold: fraction of clip_level to treat as clipped
        combine_radius: morphological closing radius for segmentation
        candidating: candidate acceptance threshold for segmentation

    Returns:
        (H, W) float32, CFA with reconstructed highlights
    """
    original = cfa.copy()

    # Pass 1: opposed inpainting
    result = reconstruct_opposed(cfa, raw_pattern, clip_levels, clip_threshold)

    # Pass 2: segmentation-based refinement (using original for refavg)
    result = reconstruct_segmented(
        result, raw_pattern, clip_levels, clip_threshold,
        combine_radius, candidating, original_cfa=original,
    )

    return result
