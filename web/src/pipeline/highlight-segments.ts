/**
 * Segmentation-based highlight reconstruction, adapted from darktable.
 *
 * Original algorithm by Iain, garagecoder (gmic team) and Hanno Schwalm (dt).
 * See: https://discuss.pixls.us/t/highlight-recovery-teaser/17670
 *
 * Works for both Bayer and X-Trans CFA patterns. The approach:
 * 1. Build 3×3 superpixel color planes at 1/3 resolution in cube-root space.
 * 2. Segment clipped regions per color plane via flood-fill.
 * 3. Optionally merge nearby segments with morphological closing.
 * 4. For each segment, find the best unclipped "candidate" pixel weighted by
 *    local smoothness and brightness.
 * 5. Inpaint clipped raw photosites by transferring pseudo-chrominance from the
 *    candidate location: output = refavg_here + (candidate − cand_reference).
 *
 * darktable is free software under GPL-3.0-or-later.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HL_BORDER = 8;
const HL_POWERF = 3.0;
const SEG_ID_MASK = 0x40000;
const MIN_SEGMENT_SIZE = 4;
const MAX_SLOTS = SEG_ID_MASK - 2; // 262142

// ---------------------------------------------------------------------------
// Segmentation data structure
// ---------------------------------------------------------------------------

interface Segmentation {
  data: Uint32Array;   // segment id per plane pixel
  tmp: Uint32Array;    // scratch buffer for morphological ops
  size: Int32Array;    // pixel count per segment
  xmin: Int32Array;
  xmax: Int32Array;
  ymin: Int32Array;
  ymax: Int32Array;
  val1: Float32Array;  // candidate value
  val2: Float32Array;  // candidate refavg
  nr: number;          // next segment index (starts at 2)
  border: number;
  slots: number;
  width: number;
  height: number;
}

function createSegmentation(
  width: number, height: number, border: number, maxSlots: number,
): Segmentation {
  const slots = Math.max(256, Math.min(maxSlots, MAX_SLOTS));
  const n = width * height;
  return {
    data: new Uint32Array(n),
    tmp: new Uint32Array(n),
    size: new Int32Array(slots),
    xmin: new Int32Array(slots),
    xmax: new Int32Array(slots),
    ymin: new Int32Array(slots),
    ymax: new Int32Array(slots),
    val1: new Float32Array(slots),
    val2: new Float32Array(slots),
    nr: 2,
    border,
    slots,
    width,
    height,
  };
}

function clearSlot(seg: Segmentation, id: number): void {
  seg.size[id] = 0;
  seg.xmin[id] = 0; seg.xmax[id] = 0;
  seg.ymin[id] = 0; seg.ymax[id] = 0;
  seg.val1[id] = 0; seg.val2[id] = 0;
}

function getSegmentId(seg: Segmentation, loc: number): number {
  if (loc >= seg.width * (seg.height - seg.border)) return 0;
  const id = seg.data[loc] & (SEG_ID_MASK - 1);
  return (id < seg.nr && id > 1) ? id : 0;
}

// ---------------------------------------------------------------------------
// Flood-fill segmentation
// ---------------------------------------------------------------------------

// Faithful port of darktable's _floodfill_segmentize from segmentation.c.
// Processes seed pixel, scans right then left, pushing above/below spans.
function floodfillSegmentize(
  yin: number, xin: number, seg: Segmentation, id: number,
  stackX: Int32Array, stackY: Int32Array,
): boolean {
  if (id >= seg.slots - 2) return false;

  const w = seg.width;
  const h = seg.height;
  const border = seg.border;
  const d = seg.data;
  const stackMax = stackX.length;
  let sp = 0;

  let minX = xin, maxX = xin, minY = yin, maxY = yin;
  let cnt = 0;
  clearSlot(seg, id);

  function updateBounds(xp: number, yp: number): void {
    minX = Math.min(minX, xp); maxX = Math.max(maxX, xp);
    minY = Math.min(minY, yp); maxY = Math.max(maxY, yp);
  }

  // push seed
  stackX[sp] = xin; stackY[sp] = yin; sp++;

  while (sp > 0) {
    sp--;
    const x = stackX[sp];
    const y = stackY[sp];
    if (d[y * w + x] !== 1) continue;

    const yUp = y - 1;
    const yDown = y + 1;
    let lastXUp = false, lastXDown = false;
    let firstXUp = false, firstXDown = false;

    // ---- Process seed pixel (x, y) ----
    d[y * w + x] = id;
    cnt++;

    // Check above at seed
    if (yUp >= border && d[yUp * w + x] === 1) {
      if (sp < stackMax) { stackX[sp] = x; stackY[sp] = yUp; sp++; }
      firstXUp = lastXUp = true;
    } else {
      if (x > border + 1 && d[yUp * w + x] === 0) {
        updateBounds(x, yUp);
        d[yUp * w + x] = SEG_ID_MASK | id;
      }
    }

    // Check below at seed
    if (yDown < h - border && d[yDown * w + x] === 1) {
      if (sp < stackMax) { stackX[sp] = x; stackY[sp] = yDown; sp++; }
      firstXDown = lastXDown = true;
    } else {
      if (yDown < h - border - 2 && d[yDown * w + x] === 0) {
        updateBounds(x, yDown);
        d[yDown * w + x] = SEG_ID_MASK | id;
      }
    }

    // ---- Scan right from x+1 ----
    let xr = x + 1;
    while (xr < w - border && d[y * w + xr] === 1) {
      d[y * w + xr] = id;
      cnt++;

      if (yUp >= border && d[yUp * w + xr] === 1) {
        if (!lastXUp && sp < stackMax) { stackX[sp] = xr; stackY[sp] = yUp; sp++; lastXUp = true; }
      } else {
        if (yUp > border + 1 && d[yUp * w + xr] === 0) {
          updateBounds(xr, yUp);
          d[yUp * w + xr] = SEG_ID_MASK | id;
        }
        lastXUp = false;
      }

      if (yDown < h - border && d[yDown * w + xr] === 1) {
        if (!lastXDown && sp < stackMax) { stackX[sp] = xr; stackY[sp] = yDown; sp++; lastXDown = true; }
      } else {
        if (yDown < h - border - 2 && d[yDown * w + xr] === 0) {
          updateBounds(xr, yDown);
          d[yDown * w + xr] = SEG_ID_MASK | id;
        }
        lastXDown = false;
      }
      xr++;
    }

    // Mark right border
    if (xr < w - border - 2 && d[y * w + xr] === 0) {
      updateBounds(xr, y);
      d[y * w + xr] = SEG_ID_MASK | id;
    }

    // ---- Scan left from x-1 ----
    let xl = x - 1;
    lastXUp = firstXUp;
    lastXDown = firstXDown;
    while (xl >= border && d[y * w + xl] === 1) {
      d[y * w + xl] = id;
      cnt++;

      if (yUp >= border && d[yUp * w + xl] === 1) {
        if (!lastXUp && sp < stackMax) { stackX[sp] = xl; stackY[sp] = yUp; sp++; lastXUp = true; }
      } else {
        if (yUp > border + 1 && d[yUp * w + xl] === 0) {
          updateBounds(xl, yUp);
          d[yUp * w + xl] = SEG_ID_MASK | id;
        }
        lastXUp = false;
      }

      if (yDown < h - border && d[yDown * w + xl] === 1) {
        if (!lastXDown && sp < stackMax) { stackX[sp] = xl; stackY[sp] = yDown; sp++; lastXDown = true; }
      } else {
        if (yDown < h - border - 2 && d[yDown * w + xl] === 0) {
          updateBounds(xl, yDown);
          d[yDown * w + xl] = SEG_ID_MASK | id;
        }
        lastXDown = false;
      }
      xl--;
    }

    // Re-mark seed (darktable does this)
    d[y * w + x] = id;

    // Mark left border
    if (xl > border + 1 && d[y * w + xl] === 0) {
      updateBounds(xl, y);
      d[y * w + xl] = SEG_ID_MASK | id;
    }
  }

  if (cnt < MIN_SEGMENT_SIZE) {
    for (let row = minY; row <= maxY; row++) {
      for (let col = minX; col <= maxX; col++) {
        const loc = row * w + col;
        if (d[loc] === id) d[loc] = 1;
        else if (d[loc] === (id | SEG_ID_MASK)) d[loc] = 0;
      }
    }
    return false;
  }

  seg.size[id] = cnt;
  seg.xmin[id] = minX; seg.xmax[id] = maxX;
  seg.ymin[id] = minY; seg.ymax[id] = maxY;
  seg.nr++;
  clearSlot(seg, id + 1);
  return true;
}

function segmentizePlane(seg: Segmentation): void {
  const { width, height, border } = seg;
  const stackSize = Math.max(1024, (width * height) >> 5);
  const stackX = new Int32Array(stackSize);
  const stackY = new Int32Array(stackSize);
  let id = 2;
  for (let row = border; row < height - border; row++) {
    for (let col = border; col < width - border; col++) {
      if (id >= seg.slots - 2) return;
      if (seg.data[row * width + col] === 1) {
        if (floodfillSegmentize(row, col, seg, id, stackX, stackY)) id++;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Morphological closing (dilate + erode)
// ---------------------------------------------------------------------------

function testDilate(img: Uint32Array, i: number, w: number, radius: number): boolean {
  // radius 1: 3×3
  if (img[i - w - 1] | img[i - w] | img[i - w + 1] |
      img[i - 1]     | img[i]     | img[i + 1] |
      img[i + w - 1] | img[i + w] | img[i + w + 1]) return true;
  if (radius < 2) return false;

  // radius 2: diamond r=2
  const w2 = 2 * w;
  if (img[i - w2 - 1] | img[i - w2] | img[i - w2 + 1] |
      img[i - w - 2]  | img[i - w + 2] |
      img[i - 2]      | img[i + 2] |
      img[i + w - 2]  | img[i + w + 2] |
      img[i + w2 - 1] | img[i + w2] | img[i + w2 + 1]) return true;
  if (radius < 3) return false;

  // radius 3
  const w3 = 3 * w;
  if (img[i - w3 - 2] | img[i - w3 - 1] | img[i - w3] | img[i - w3 + 1] | img[i - w3 + 2] |
      img[i - w2 - 3] | img[i - w2 - 2] | img[i - w2 + 2] | img[i - w2 + 3] |
      img[i - w - 3]  | img[i - w + 3] |
      img[i - 3]      | img[i + 3] |
      img[i + w - 3]  | img[i + w + 3] |
      img[i + w2 - 3] | img[i + w2 - 2] | img[i + w2 + 2] | img[i + w2 + 3] |
      img[i + w3 - 2] | img[i + w3 - 1] | img[i + w3] | img[i + w3 + 1] | img[i + w3 + 2]) return true;
  if (radius < 4) return false;

  // radius 4
  const w4 = 4 * w;
  if (img[i - w4 - 2] | img[i - w4 - 1] | img[i - w4] | img[i - w4 + 1] | img[i - w4 + 2] |
      img[i - w3 - 3] | img[i - w3 + 3] |
      img[i - w2 - 4] | img[i - w2 + 4] |
      img[i - w - 4]  | img[i - w + 4] |
      img[i - 4]      | img[i + 4] |
      img[i + w - 4]  | img[i + w + 4] |
      img[i + w2 - 4] | img[i + w2 + 4] |
      img[i + w3 - 3] | img[i + w3 + 3] |
      img[i + w4 - 2] | img[i + w4 - 1] | img[i + w4] | img[i + w4 + 1] | img[i + w4 + 2]) return true;
  if (radius < 5) return false;

  // radius 5
  const w5 = 5 * w;
  if (img[i - w5 - 2] | img[i - w5 - 1] | img[i - w5] | img[i - w5 + 1] | img[i - w5 + 2] |
      img[i - w4 - 4] | img[i - w4 - 3] | img[i - w4 + 3] | img[i - w4 + 4] |
      img[i - w3 - 4] | img[i - w3 + 4] |
      img[i - w2 - 5] | img[i - w2 + 5] |
      img[i - w - 5]  | img[i - w + 5] |
      img[i - 5]      | img[i + 5] |
      img[i + w - 5]  | img[i + w + 5] |
      img[i + w2 - 5] | img[i + w2 + 5] |
      img[i + w3 - 4] | img[i + w3 + 4] |
      img[i + w4 - 4] | img[i + w4 - 3] | img[i + w4 + 3] | img[i + w4 + 4] |
      img[i + w5 - 2] | img[i + w5 - 1] | img[i + w5] | img[i + w5 + 1] | img[i + w5 + 2]) return true;

  return false;
}

function testErode(img: Uint32Array, i: number, w: number, radius: number): boolean {
  if (!(img[i - w - 1] & img[i - w] & img[i - w + 1] &
        img[i - 1]     & img[i]     & img[i + 1] &
        img[i + w - 1] & img[i + w] & img[i + w + 1])) return false;
  if (radius < 2) return true;

  const w2 = 2 * w;
  if (!(img[i - w2 - 1] & img[i - w2] & img[i - w2 + 1] &
        img[i - w - 2]  & img[i - w + 2] &
        img[i - 2]      & img[i + 2] &
        img[i + w - 2]  & img[i + w + 2] &
        img[i + w2 - 1] & img[i + w2] & img[i + w2 + 1])) return false;
  if (radius < 3) return true;

  const w3 = 3 * w;
  if (!(img[i - w3 - 2] & img[i - w3 - 1] & img[i - w3] & img[i - w3 + 1] & img[i - w3 + 2] &
        img[i - w2 - 3] & img[i - w2 - 2] & img[i - w2 + 2] & img[i - w2 + 3] &
        img[i - w - 3]  & img[i - w + 3] &
        img[i - 3]      & img[i + 3] &
        img[i + w - 3]  & img[i + w + 3] &
        img[i + w2 - 3] & img[i + w2 - 2] & img[i + w2 + 2] & img[i + w2 + 3] &
        img[i + w3 - 2] & img[i + w3 - 1] & img[i + w3] & img[i + w3 + 1] & img[i + w3 + 2])) return false;
  if (radius < 4) return true;

  const w4 = 4 * w;
  if (!(img[i - w4 - 2] & img[i - w4 - 1] & img[i - w4] & img[i - w4 + 1] & img[i - w4 + 2] &
        img[i - w3 - 3] & img[i - w3 + 3] &
        img[i - w2 - 4] & img[i - w2 + 4] &
        img[i - w - 4]  & img[i - w + 4] &
        img[i - 4]      & img[i + 4] &
        img[i + w - 4]  & img[i + w + 4] &
        img[i + w2 - 4] & img[i + w2 + 4] &
        img[i + w3 - 3] & img[i + w3 + 3] &
        img[i + w4 - 2] & img[i + w4 - 1] & img[i + w4] & img[i + w4 + 1] & img[i + w4 + 2])) return false;
  if (radius < 5) return true;

  const w5 = 5 * w;
  return !!(img[i - w5 - 2] & img[i - w5 - 1] & img[i - w5] & img[i - w5 + 1] & img[i - w5 + 2] &
        img[i - w4 - 4] & img[i - w4 - 3] & img[i - w4 + 3] & img[i - w4 + 4] &
        img[i - w3 - 4] & img[i - w3 + 4] &
        img[i - w2 - 5] & img[i - w2 + 5] &
        img[i - w - 5]  & img[i - w + 5] &
        img[i - 5]      & img[i + 5] &
        img[i + w - 5]  & img[i + w + 5] &
        img[i + w2 - 5] & img[i + w2 + 5] &
        img[i + w3 - 4] & img[i + w3 + 4] &
        img[i + w4 - 4] & img[i + w4 - 3] & img[i + w4 + 3] & img[i + w4 + 4] &
        img[i + w5 - 2] & img[i + w5 - 1] & img[i + w5] & img[i + w5 + 1] & img[i + w5 + 2]);
}

function dilate(src: Uint32Array, dst: Uint32Array, w: number, h: number, border: number, radius: number): void {
  for (let row = border; row < h - border; row++) {
    for (let col = border; col < w - border; col++) {
      const i = row * w + col;
      dst[i] = testDilate(src, i, w, radius) ? 1 : 0;
    }
  }
}

function erode(src: Uint32Array, dst: Uint32Array, w: number, h: number, border: number, radius: number): void {
  for (let row = border; row < h - border; row++) {
    for (let col = border; col < w - border; col++) {
      const i = row * w + col;
      dst[i] = testErode(src, i, w, radius) ? 1 : 0;
    }
  }
}

function fillBorder(d: Uint32Array, w: number, h: number, val: number, border: number): void {
  // Top and bottom bands
  const topEnd = border * w;
  const botStart = (h - border) * w;
  for (let i = 0; i < topEnd; i++) d[i] = val;
  for (let i = botStart; i < w * h; i++) d[i] = val;
  // Left and right columns
  for (let row = border; row < h - border; row++) {
    const base = row * w;
    for (let i = 0; i < border; i++) {
      d[base + i] = val;
      d[base + w - 1 - i] = val;
    }
  }
}

function segmentsCombine(seg: Segmentation, radius: number): void {
  if (radius <= 0) return;
  const { data, tmp, width, height, border } = seg;
  fillBorder(data, width, height, 0, border);
  dilate(data, tmp, width, height, border, radius);
  if (radius > 3) {
    fillBorder(tmp, width, height, 1, border);
    erode(tmp, data, width, height, border, radius - 3);
  } else {
    data.set(tmp);
  }
  fillBorder(data, width, height, 0, border);
}

// ---------------------------------------------------------------------------
// Candidate selection (weight-based)
// ---------------------------------------------------------------------------

function localStdDev(p: Float32Array, idx: number, w: number): number {
  const w2 = 2 * w;
  // 21-tap cross-shaped kernel (5×5 minus corners of corners)
  const vals = [
    p[idx - w2 - 1], p[idx - w2], p[idx - w2 + 1],
    p[idx - w - 2], p[idx - w - 1], p[idx - w], p[idx - w + 1], p[idx - w + 2],
    p[idx - 2], p[idx - 1], p[idx], p[idx + 1], p[idx + 2],
    p[idx + w - 2], p[idx + w - 1], p[idx + w], p[idx + w + 1], p[idx + w + 2],
    p[idx + w2 - 1], p[idx + w2], p[idx + w2 + 1],
  ];
  let sum = 0;
  for (let i = 0; i < 21; i++) sum += vals[i];
  const av = sum / 21;
  let variance = 0;
  for (let i = 0; i < 21; i++) { const d = vals[i] - av; variance += d * d; }
  return Math.sqrt(variance / 21);
}

function calcWeight(plane: Float32Array, loc: number, w: number, clipval: number): number {
  const smoothness = Math.max(0, 1 - 10 * Math.sqrt(localStdDev(plane, loc, w)));
  // 3×3 average
  let val = 0;
  for (let y = -1; y <= 1; y++) {
    for (let x = -1; x <= 1; x++) {
      val += plane[loc + y * w + x];
    }
  }
  val /= 9;
  const sval = Math.max(1, (Math.min(clipval, val) / clipval) ** 2);
  return sval * smoothness;
}

// Gaussian-weighted 5×5 kernel for candidate value averaging
const GAUSS_5X5 = [
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, 36, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1],
];

function calcPlaneCandidates(
  plane: Float32Array, refavg: Float32Array, seg: Segmentation,
  clipval: number, badlevel: number,
): void {
  for (let id = 2; id < seg.nr; id++) {
    seg.val1[id] = 0;
    seg.val2[id] = 0;

    if ((seg.ymax[id] - seg.ymin[id] <= 2) || (seg.xmax[id] - seg.xmin[id] <= 2))
      continue;

    let testref = -1;
    let testweight = 0;

    const rowMin = Math.max(seg.border + 2, seg.ymin[id] - 2);
    const rowMax = Math.min(seg.height - seg.border - 2, seg.ymax[id] + 3);
    const colMin = Math.max(seg.border + 2, seg.xmin[id] - 2);
    const colMax = Math.min(seg.width - seg.border - 2, seg.xmax[id] + 3);

    for (let row = rowMin; row < rowMax; row++) {
      for (let col = colMin; col < colMax; col++) {
        const pos = row * seg.width + col;
        const sid = getSegmentId(seg, pos);
        if (sid === id && plane[pos] < clipval) {
          const isBorder = (seg.data[pos] & SEG_ID_MASK) ? 1.0 : 0.75;
          const wht = calcWeight(plane, pos, seg.width, clipval) * isBorder;
          if (wht > testweight) {
            testweight = wht;
            testref = pos;
          }
        }
      }
    }

    if (testref >= 0 && testweight > 1.0 - badlevel) {
      let sum = 0, pix = 0;
      for (let y = -2; y <= 2; y++) {
        for (let x = -2; x <= 2; x++) {
          const pos = testref + y * seg.width + x;
          const w = GAUSS_5X5[y + 2][x + 2];
          if (plane[pos] < clipval) {
            sum += plane[pos] * w;
            pix += w;
          }
        }
      }
      const av = sum / Math.max(1, pix);
      if (av > 0.125 * clipval) {
        seg.val1[id] = Math.min(clipval, av);
        seg.val2[id] = refavg[testref];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Plane helpers
// ---------------------------------------------------------------------------

function rawToPlane(pwidth: number, row: number, col: number): number {
  return (HL_BORDER + Math.floor(row / 3)) * pwidth + Math.floor(col / 3) + HL_BORDER;
}

function extendBorder(mask: Float32Array, width: number, height: number, border: number): void {
  if (border <= 0) return;
  // Extend rows horizontally
  for (let row = border; row < height - border; row++) {
    const base = row * width;
    for (let i = 0; i < border; i++) {
      mask[base + i] = mask[base + border];
      mask[base + width - 1 - i] = mask[base + width - border - 1];
    }
  }
  // Extend columns vertically
  for (let col = 0; col < width; col++) {
    const clamped = Math.min(width - border - 1, Math.max(col, border));
    const top = mask[border * width + clamped];
    const bot = mask[(height - border - 1) * width + clamped];
    for (let i = 0; i < border; i++) {
      mask[i * width + col] = top;
      mask[(height - 1 - i) * width + col] = bot;
    }
  }
}

// ---------------------------------------------------------------------------
// Refavg calculation (cube-root opposed channel average)
// ---------------------------------------------------------------------------

function calcRefavg(
  cfa: Float32Array, width: number, height: number,
  row: number, col: number, ch: number,
  getCh: (y: number, x: number) => number,
): number {
  const mean = [0, 0, 0];
  const cnt = [0, 0, 0];

  const yMin = Math.max(0, row - 1);
  const yMax = Math.min(height - 1, row + 1);
  const xMin = Math.max(0, col - 1);
  const xMax = Math.min(width - 1, col + 1);

  for (let ny = yMin; ny <= yMax; ny++) {
    for (let nx = xMin; nx <= xMax; nx++) {
      const val = Math.max(0, cfa[ny * width + nx]);
      const c = getCh(ny, nx);
      mean[c] += val;
      cnt[c]++;
    }
  }

  const cr = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    cr[c] = cnt[c] > 0 ? Math.cbrt(mean[c] / cnt[c]) : 0;
  }

  let oppCr: number;
  if (ch === 0) oppCr = 0.5 * (cr[1] + cr[2]);
  else if (ch === 1) oppCr = 0.5 * (cr[0] + cr[2]);
  else oppCr = 0.5 * (cr[0] + cr[1]);

  return oppCr; // in cube-root space (caller can cube if needed)
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * Segmentation-based highlight reconstruction for CFA raw data.
 *
 * Operates in-place on the `cfa` array, extending clipped pixel values using
 * segment-based candidate selection. This is a more sophisticated alternative
 * to the opposed-inpainting approach that better handles structured highlights
 * by analyzing clipped regions as coherent segments.
 *
 * @param combineRadius - morphological closing radius for merging nearby
 *   segments (0–5, default 2). Higher values merge more aggressively.
 * @param candidating - badlevel threshold for candidate acceptance (0–1,
 *   default 0.5). Lower values are stricter.
 */
/**
 * @param originalCfa - If provided, used for per-pixel refavg in the final
 *   reconstruction step (matching darktable's behavior of using the original
 *   raw input, not the opposed-inpainted version). When omitted, `cfa` is
 *   used for both plane building and refavg.
 */
export function reconstructHighlightsSegmented(
  cfa: Float32Array, width: number, height: number,
  pattern: readonly (readonly number[])[], period: number,
  dy: number, dx: number,
  combineRadius = 2,
  candidating = 0.5,
  originalCfa?: Float32Array,
): void {
  const clip = 1.0;
  const original = originalCfa ?? cfa;

  function getCh(y: number, x: number): number {
    return pattern[((y + dy) % period + period) % period][((x + dx) % period + period) % period];
  }

  // ---- Step 1: Build downsampled color planes in cube-root space ----

  // Superpixel alignment: for Bayer with green at (0,0), center on col%3==1;
  // otherwise col%3==2.  This ensures the 3×3 box has green in the centre
  // giving a 5:2:2 G:R:B ratio for better chroma stability.
  const isGreenAt00 = period === 2 && getCh(0, 0) === 1;
  const xshifter = isGreenAt00 ? 1 : 2;

  // Round up to even, matching darktable's dt_round_size(n, 2)
  const roundEven = (n: number) => (n + 1) & ~1;
  const pwidth = roundEven(Math.floor(width / 3)) + 2 * HL_BORDER;
  const pheight = roundEven(Math.floor(height / 3)) + 2 * HL_BORDER;
  const psize = pwidth * pheight;

  // 3 color planes + 3 refavg planes
  const planes: Float32Array[] = [];
  const refavgs: Float32Array[] = [];
  for (let c = 0; c < 3; c++) {
    planes.push(new Float32Array(psize));
    refavgs.push(new Float32Array(psize));
  }

  // Cube-root clip thresholds per channel (before WB, all channels clip at 1.0)
  const cubeClip = Math.cbrt(clip);

  // Segmentation structures: one per color plane
  const maxSegments = Math.max(256, Math.floor((width * height) / 4000));
  const segs: Segmentation[] = [];
  for (let c = 0; c < 3; c++) {
    segs.push(createSegmentation(pwidth, pheight, HL_BORDER + 1, maxSegments));
  }

  // Populate planes from 3×3 superpixels
  let anyClipped = 0;
  for (let row = 1; row < height - 1; row++) {
    for (let col = 1; col < width - 1; col++) {
      if (col % 3 !== xshifter || row % 3 !== 1) continue;

      const mean = [0, 0, 0];
      const cnt = [0, 0, 0];

      for (let sdy = row - 1; sdy < row + 2; sdy++) {
        for (let sdx = col - 1; sdx < col + 2; sdx++) {
          const val = cfa[sdy * width + sdx];
          const c = getCh(sdy, sdx);
          mean[c] += val;
          cnt[c]++;
        }
      }

      // Convert to cube-root space
      for (let c = 0; c < 3; c++) {
        mean[c] = cnt[c] > 0 ? Math.cbrt(mean[c] / cnt[c]) : 0;
      }

      const cubeRefavg = [
        0.5 * (mean[1] + mean[2]), // opposed for R
        0.5 * (mean[0] + mean[2]), // opposed for G
        0.5 * (mean[0] + mean[1]), // opposed for B
      ];

      const o = rawToPlane(pwidth, row, col);

      for (let c = 0; c < 3; c++) {
        planes[c][o] = mean[c];
        refavgs[c][o] = cubeRefavg[c];

        if (mean[c] > cubeClip) {
          segs[c].data[o] = 1; // mark as clipped
          anyClipped++;
        }
      }
    }
  }

  if (anyClipped < 20) return; // nothing significant to reconstruct

  // ---- Step 2: Extend border data ----

  for (let c = 0; c < 3; c++) {
    extendBorder(planes[c], pwidth, pheight, HL_BORDER);
  }

  // ---- Step 3: Morphological closing + segmentation ----

  for (let c = 0; c < 3; c++) {
    segmentsCombine(segs[c], combineRadius);
    segmentizePlane(segs[c]);
  }

  // ---- Step 4: Find best candidates per segment ----

  for (let c = 0; c < 3; c++) {
    calcPlaneCandidates(planes[c], refavgs[c], segs[c], cubeClip, candidating);
  }

  // ---- Step 5: Reconstruct clipped raw pixels ----
  // Use original (pre-opposed) values for threshold and max — matching
  // darktable which uses the raw input, not the opposed output.

  for (let row = 1; row < height - 1; row++) {
    for (let col = 1; col < width - 1; col++) {
      const idx = row * width + col;
      const inval = Math.max(0, original[idx]);
      if (inval < clip) continue;

      const color = getCh(row, col);
      const o = rawToPlane(pwidth, row, col);
      const pid = getSegmentId(segs[color], o);

      if (pid > 1 && pid < segs[color].nr) {
        const candidate = segs[color].val1[pid];
        if (candidate !== 0) {
          const candReference = segs[color].val2[pid];
          const refavgHere = calcRefavg(original, width, height, row, col, color, getCh);
          const oval = (refavgHere + candidate - candReference) ** HL_POWERF;
          cfa[idx] = Math.max(inval, oval);
        }
      }
    }
  }
}

/** Debug variant that returns diagnostics. */
export function _reconstructDebug(
  cfa: Float32Array, width: number, height: number,
  pattern: readonly (readonly number[])[], period: number,
  dy: number, dx: number,
  combineRadius = 2, candidating = 0.5, originalCfa?: Float32Array,
): { anyClipped: number; segCounts: number[]; candidates: { id: number; val1: number; val2: number }[][] } {
  const clip = 1.0;
  const original = originalCfa ?? cfa;
  function getCh(y: number, x: number): number {
    return pattern[((y + dy) % period + period) % period][((x + dx) % period + period) % period];
  }
  const isGreenAt00 = period === 2 && getCh(0, 0) === 1;
  const xshifter = isGreenAt00 ? 1 : 2;
  const roundEven = (n: number) => (n + 1) & ~1;
  const pwidth = roundEven(Math.floor(width / 3)) + 2 * HL_BORDER;
  const pheight = roundEven(Math.floor(height / 3)) + 2 * HL_BORDER;
  const psize = pwidth * pheight;
  const planes: Float32Array[] = [];
  const refavgs: Float32Array[] = [];
  for (let c = 0; c < 3; c++) { planes.push(new Float32Array(psize)); refavgs.push(new Float32Array(psize)); }
  const cubeClip = Math.cbrt(clip);
  const maxSegments = Math.max(256, Math.floor((width * height) / 4000));
  const segs: Segmentation[] = [];
  for (let c = 0; c < 3; c++) segs.push(createSegmentation(pwidth, pheight, HL_BORDER + 1, maxSegments));

  let anyClipped = 0;
  for (let row = 1; row < height - 1; row++) {
    for (let col = 1; col < width - 1; col++) {
      if (col % 3 !== xshifter || row % 3 !== 1) continue;
      const mean = [0, 0, 0]; const cnt = [0, 0, 0];
      for (let sdy = row - 1; sdy < row + 2; sdy++)
        for (let sdx = col - 1; sdx < col + 2; sdx++) {
          const c = getCh(sdy, sdx); mean[c] += cfa[sdy * width + sdx]; cnt[c]++;
        }
      for (let c = 0; c < 3; c++) mean[c] = cnt[c] > 0 ? Math.cbrt(mean[c] / cnt[c]) : 0;
      const cr = [0.5*(mean[1]+mean[2]), 0.5*(mean[0]+mean[2]), 0.5*(mean[0]+mean[1])];
      const o = rawToPlane(pwidth, row, col);
      for (let c = 0; c < 3; c++) {
        planes[c][o] = mean[c]; refavgs[c][o] = cr[c];
        if (mean[c] > cubeClip) { segs[c].data[o] = 1; anyClipped++; }
      }
    }
  }
  for (let c = 0; c < 3; c++) extendBorder(planes[c], pwidth, pheight, HL_BORDER);
  for (let c = 0; c < 3; c++) { segmentsCombine(segs[c], combineRadius); segmentizePlane(segs[c]); }
  for (let c = 0; c < 3; c++) calcPlaneCandidates(planes[c], refavgs[c], segs[c], cubeClip, candidating);

  const segCounts = segs.map(s => s.nr - 2);
  const candidates = segs.map(s => {
    const out: { id: number; val1: number; val2: number }[] = [];
    for (let id = 2; id < s.nr; id++) if (s.val1[id] !== 0) out.push({ id, val1: s.val1[id], val2: s.val2[id] });
    return out;
  });

  // Also do the reconstruction (use original for inval, matching main function)
  for (let row = 1; row < height - 1; row++) {
    for (let col = 1; col < width - 1; col++) {
      const idx = row * width + col;
      const inval = Math.max(0, original[idx]);
      if (inval < clip) continue;
      const color = getCh(row, col);
      const o = rawToPlane(pwidth, row, col);
      const pid = getSegmentId(segs[color], o);
      if (pid > 1 && pid < segs[color].nr) {
        const candidate = segs[color].val1[pid];
        if (candidate !== 0) {
          const candRef = segs[color].val2[pid];
          const ra = calcRefavg(original, width, height, row, col, color, getCh);
          cfa[idx] = Math.max(inval, (ra + candidate - candRef) ** HL_POWERF);
        }
      }
    }
  }

  return { anyClipped, segCounts, candidates };
}
