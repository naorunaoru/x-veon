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
# CFA pattern helpers
# ---------------------------------------------------------------------------

def _make_get_ch(raw_pattern: np.ndarray):
    """Return a function getCh(y, x) -> channel index (0=R, 1=G, 2=B)."""
    pat = raw_pattern.copy()
    pat[pat == 3] = 1  # G2 -> G
    h, w = pat.shape

    def get_ch(y: int, x: int) -> int:
        return int(pat[y % h, x % w])

    return get_ch


# ---------------------------------------------------------------------------
# Pass 1: Inpaint-opposed
# ---------------------------------------------------------------------------

def _calc_refavg_linear(
    cfa: np.ndarray, h: int, w: int,
    y: int, x: int, ch: int, get_ch,
) -> float:
    """Opposed-channel reference average in cube-root space, returned as linear."""
    mean = [0.0, 0.0, 0.0]
    cnt = [0, 0, 0]
    y0, y1 = max(0, y - 1), min(h - 1, y + 1)
    x0, x1 = max(0, x - 1), min(w - 1, x + 1)
    for ny in range(y0, y1 + 1):
        for nx in range(x0, x1 + 1):
            val = max(0.0, float(cfa[ny, nx]))
            c = get_ch(ny, nx)
            mean[c] += val
            cnt[c] += 1
    cr = [0.0, 0.0, 0.0]
    for c in range(3):
        cr[c] = np.cbrt(mean[c] / cnt[c]) if cnt[c] > 0 else 0.0
    if ch == 0:
        opp = 0.5 * (cr[1] + cr[2])
    elif ch == 1:
        opp = 0.5 * (cr[0] + cr[2])
    else:
        opp = 0.5 * (cr[0] + cr[1])
    return opp ** HL_POWERF


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
    get_ch = _make_get_ch(raw_pattern)

    clips = clip_levels * clip_threshold
    lo_clips = 0.2 * clips

    out = cfa.copy()

    # Step 1: Build per-channel clip mask at 1/3 resolution (superpixel level)
    mwidth = w // 3
    mheight = h // 3
    # 3 planes: per-channel clip mask
    mask = np.zeros((3, mheight, mwidth), dtype=np.uint8)

    any_clipped = False
    for mrow in range(1, mheight - 1):
        for mcol in range(1, mwidth - 1):
            # Check 3x3 raw pixels centered on this superpixel
            cy, cx = 3 * mrow, 3 * mcol
            mbuff = [0, 0, 0]
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ry, rx = cy + dy, cx + dx
                    if 0 <= ry < h and 0 <= rx < w:
                        color = get_ch(ry, rx)
                        if cfa[ry, rx] >= clips[color]:
                            mbuff[color] += 1
            for c in range(3):
                if mbuff[c] > 0:
                    mask[c, mrow, mcol] = 1
                    any_clipped = True

    if not any_clipped:
        return out

    # Step 2: Dilate mask ~3 superpixels to find nearby unclipped pixels
    import cv2
    dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = np.zeros_like(mask)
    for c in range(3):
        dilated[c] = cv2.dilate(mask[c], dil_kern)

    # Step 3: Chrominance from unclipped pixels within dilated mask
    chrom_sum = [0.0, 0.0, 0.0]
    chrom_cnt = [0, 0, 0]
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            color = get_ch(y, x)
            inval = float(cfa[y, x])
            if inval >= clips[color] or inval <= lo_clips[color]:
                continue
            # Check if this pixel is within dilated mask for its channel
            my, mx = y // 3, x // 3
            if my >= mheight or mx >= mwidth:
                continue
            if not dilated[color, my, mx]:
                continue
            ref = _calc_refavg_linear(cfa, h, w, y, x, color, get_ch)
            chrom_sum[color] += inval - ref
            chrom_cnt[color] += 1

    chrom = [0.0, 0.0, 0.0]
    for c in range(3):
        if chrom_cnt[c] > 100:
            chrom[c] = chrom_sum[c] / chrom_cnt[c]

    # Step 4: Extend clipped pixels
    n_fixed = 0
    for y in range(0, h):
        for x in range(0, w):
            color = get_ch(y, x)
            inval = max(0.0, float(cfa[y, x]))
            if inval < clips[color]:
                continue
            ref = _calc_refavg_linear(cfa, h, w, y, x, color, get_ch)
            out[y, x] = max(inval, ref + chrom[color])
            n_fixed += 1

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
    """Morphological closing (dilate then erode) on the segmentation mask."""
    if radius <= 0:
        return

    w, h, border = seg.width, seg.height, seg.border
    d, tmp = seg.data, seg.tmp

    # Fill borders with 0
    for row in range(h):
        base = row * w
        if row < border or row >= h - border:
            d[base:base + w] = 0
        else:
            d[base:base + border] = 0
            d[base + w - border:base + w] = 0

    # Dilate: any neighbor within radius has value -> set to 1
    tmp[:] = 0
    for row in range(border, h - border):
        for col in range(border, w - border):
            i = row * w + col
            if d[i]:
                tmp[i] = 1
                continue
            # Check neighborhood
            found = False
            for dy in range(-min(radius, row - border), min(radius, h - border - 1 - row) + 1):
                if found:
                    break
                for dx in range(-min(radius, col - border), min(radius, w - border - 1 - col) + 1):
                    if dy * dy + dx * dx <= radius * radius:
                        if d[(row + dy) * w + col + dx]:
                            found = True
                            break
            tmp[i] = 1 if found else 0

    # Erode: all neighbors within (radius-1) must be set
    erode_r = max(radius - 1, 1)
    for row in range(border, h - border):
        for col in range(border, w - border):
            i = row * w + col
            if not tmp[i]:
                d[i] = 0
                continue
            all_set = True
            for dy in range(-min(erode_r, row - border), min(erode_r, h - border - 1 - row) + 1):
                if not all_set:
                    break
                for dx in range(-min(erode_r, col - border), min(erode_r, w - border - 1 - col) + 1):
                    if dy * dy + dx * dx <= erode_r * erode_r:
                        if not tmp[(row + dy) * w + col + dx]:
                            all_set = False
                            break
            d[i] = 1 if all_set else 0


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


def _raw_to_plane(pwidth: int, row: int, col: int) -> int:
    return (HL_BORDER + row // 3) * pwidth + col // 3 + HL_BORDER


def _extend_border(arr: np.ndarray, width: int, height: int, border: int):
    """Extend plane data to fill border region."""
    if border <= 0:
        return
    for row in range(border, height - border):
        base = row * width
        for i in range(border):
            arr[base + i] = arr[base + border]
            arr[base + width - 1 - i] = arr[base + width - border - 1]
    for col in range(width):
        clamped = min(width - border - 1, max(col, border))
        top_val = arr[border * width + clamped]
        bot_val = arr[(height - border - 1) * width + clamped]
        for i in range(border):
            arr[i * width + col] = top_val
            arr[(height - 1 - i) * width + col] = bot_val


def _calc_refavg_at(
    cfa: np.ndarray, width: int, height: int,
    row: int, col: int, ch: int, get_ch,
) -> float:
    """Cube-root opposed channel average for a single CFA pixel."""
    mean = [0.0, 0.0, 0.0]
    cnt = [0, 0, 0]
    y0, y1 = max(0, row - 1), min(height - 1, row + 1)
    x0, x1 = max(0, col - 1), min(width - 1, col + 1)
    for ny in range(y0, y1 + 1):
        for nx in range(x0, x1 + 1):
            val = max(0.0, float(cfa[ny * width + nx]))
            c = get_ch(ny, nx)
            mean[c] += val
            cnt[c] += 1
    cr = [0.0, 0.0, 0.0]
    for c in range(3):
        cr[c] = np.cbrt(mean[c] / cnt[c]) if cnt[c] > 0 else 0.0
    if ch == 0:
        return 0.5 * (cr[1] + cr[2])
    elif ch == 1:
        return 0.5 * (cr[0] + cr[2])
    else:
        return 0.5 * (cr[0] + cr[1])


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
    get_ch = _make_get_ch(raw_pattern)
    pat = raw_pattern.copy()
    pat[pat == 3] = 1

    clips = clip_levels * clip_threshold

    # Determine superpixel alignment
    # For Bayer with green at (0,0), center on col%3==1; otherwise col%3==2
    period_h, period_w = pat.shape
    is_bayer = (period_h <= 2 and period_w <= 2)
    is_green_00 = is_bayer and get_ch(0, 0) == 1
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

    # Step 1: Build downsampled color planes from 3x3 superpixels
    any_clipped = 0
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            if col % 3 != xshifter or row % 3 != 1:
                continue

            mean = [0.0, 0.0, 0.0]
            cnt = [0, 0, 0]
            for sdy in range(row - 1, row + 2):
                for sdx in range(col - 1, col + 2):
                    val = float(cfa[sdy, sdx])
                    c = get_ch(sdy, sdx)
                    mean[c] += val
                    cnt[c] += 1

            for c in range(3):
                mean[c] = np.cbrt(mean[c] / cnt[c]) if cnt[c] > 0 else 0.0

            cube_refavg = [
                0.5 * (mean[1] + mean[2]),
                0.5 * (mean[0] + mean[2]),
                0.5 * (mean[0] + mean[1]),
            ]

            o = _raw_to_plane(pwidth, row, col)
            for c in range(3):
                planes[c][o] = mean[c]
                refavgs[c][o] = cube_refavg[c]
                if mean[c] >= cube_clips[c]:
                    segs[c].data[o] = 1
                    any_clipped += 1

    if any_clipped < 20:
        return cfa.copy()

    # Step 2: Extend border data
    for c in range(3):
        _extend_border(planes[c], pwidth, pheight, HL_BORDER)

    # Step 3: Morphological closing + segmentation
    for c in range(3):
        _morphological_close(segs[c], combine_radius)
        _segmentize_plane(segs[c])

    # Step 4: Find best candidates per segment
    for c in range(3):
        _calc_plane_candidates(planes[c], refavgs[c], segs[c], cube_clips[c], candidating)

    # Step 5: Reconstruct clipped raw pixels
    out = cfa.copy()
    n_fixed = 0
    original_flat = original.ravel()

    for row in range(1, h - 1):
        for col in range(1, w - 1):
            idx = row * w + col
            inval = max(0.0, float(original_flat[idx]))
            color = get_ch(row, col)
            if inval < clips[color]:
                continue

            o = _raw_to_plane(pwidth, row, col)
            pid = _get_seg_id(segs[color], o)

            if 1 < pid < segs[color].nr:
                candidate = float(segs[color].val1[pid])
                if candidate != 0:
                    cand_ref = float(segs[color].val2[pid])
                    refavg_here = _calc_refavg_at(
                        original_flat, w, h, row, col, color, get_ch,
                    )
                    oval = (refavg_here + candidate - cand_ref) ** HL_POWERF
                    out[row, col] = max(inval, oval)
                    n_fixed += 1

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
