import { XTRANS_PATTERN } from './constants.js';

/**
 * Crop raw sensor data to the visible area using rawloader's crop values.
 * rawloader returns the full sensor including optical black borders;
 * rawpy.raw_image_visible does this crop automatically.
 *
 * @param {Uint16Array} rawData - Full sensor pixels
 * @param {number} fullWidth - Full sensor width
 * @param {number} fullHeight - Full sensor height
 * @param {Uint16Array} crops - [top, right, bottom, left] pixels to remove
 * @returns {{ data: Uint16Array, width: number, height: number }}
 */
export function cropToVisible(rawData, fullWidth, fullHeight, crops) {
  const top = crops[0], right = crops[1], bottom = crops[2], left = crops[3];
  const visW = fullWidth - left - right;
  const visH = fullHeight - top - bottom;

  if (top === 0 && right === 0 && bottom === 0 && left === 0) {
    return { data: rawData, width: fullWidth, height: fullHeight };
  }

  const out = new Uint16Array(visW * visH);
  for (let y = 0; y < visH; y++) {
    const srcOffset = (y + top) * fullWidth + left;
    out.set(rawData.subarray(srcOffset, srcOffset + visW), y * visW);
  }

  return { data: out, width: visW, height: visH };
}

/**
 * Find the CFA pattern shift using rawloader's CFA metadata.
 *
 * Returns (dy, dx) in additive convention:
 *   XTRANS_PATTERN[(y + dy) % 6][(x + dx) % 6] = visible_channel(y, x)
 *
 * This matches the visible CFA against the reference X-Trans pattern by
 * parsing the CFA string from rawloader and applying the crop offset.
 *
 * @param {string} cfaStr - 36-character CFA string from rawloader (e.g. "RBGBRG...")
 * @param {number} cfaWidth - CFA pattern width (must be 6 for X-Trans)
 * @param {Uint16Array} crops - [top, right, bottom, left] crop values
 * @returns {{ dy: number, dx: number }}
 */
export function findPatternShift(cfaStr, cfaWidth, crops) {
  if (cfaStr.length !== 36 || cfaWidth !== 6) {
    throw new Error(`Unsupported CFA: ${cfaStr.length} chars, width ${cfaWidth}. Only 6x6 X-Trans supported.`);
  }

  const top = crops[0], left = crops[3];

  // Build visible-area 6x6 CFA pattern (after crop offset)
  const vis = [];
  for (let y = 0; y < 6; y++) {
    vis[y] = [];
    for (let x = 0; x < 6; x++) {
      const srcY = (y + top) % 6;
      const srcX = (x + left) % 6;
      const ch = cfaStr[srcY * 6 + srcX];
      vis[y][x] = ch === 'R' ? 0 : ch === 'G' ? 1 : 2;
    }
  }

  // Find (dy, dx) such that ref[(y+dy)%6][(x+dx)%6] = vis[y][x]
  for (let dy = 0; dy < 6; dy++) {
    for (let dx = 0; dx < 6; dx++) {
      let match = true;
      for (let y = 0; y < 6 && match; y++) {
        for (let x = 0; x < 6 && match; x++) {
          if (XTRANS_PATTERN[(y + dy) % 6][(x + dx) % 6] !== vis[y][x]) {
            match = false;
          }
        }
      }
      if (match) return { dy, dx };
    }
  }

  throw new Error(`CFA pattern does not match X-Trans reference: ${cfaStr}`);
}

/**
 * Normalize raw CFA data to [0, 1] float32.
 *
 * @param {Uint16Array} rawData - Raw sensor pixels
 * @param {number} width
 * @param {number} height
 * @param {Uint16Array} blackLevels - Per-channel black levels
 * @param {Uint16Array} whiteLevels - Per-channel white levels
 * @returns {Float32Array}
 */
export function normalizeRawCfa(rawData, width, height, blackLevels, whiteLevels) {
  const black = blackLevels[0];
  const white = whiteLevels[0];
  const range = white - black;
  const n = width * height;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (rawData[i] - black) / range;
  }
  return out;
}

/**
 * Apply white balance to normalized CFA data (in-place).
 * Each pixel is multiplied by the WB coefficient of its CFA channel.
 *
 * @param {Float32Array} cfa - Normalized CFA data (mutated in-place)
 * @param {number} width
 * @param {number} height
 * @param {Float32Array} wb - [R, G, B] WB coefficients (G-normalized)
 * @param {number} dy - Pattern shift Y
 * @param {number} dx - Pattern shift X
 */
export function applyWhiteBalance(cfa, width, height, wb, dy, dx) {
  for (let y = 0; y < height; y++) {
    const patY = (y + dy) % 6;
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const ch = XTRANS_PATTERN[patY][(x + dx) % 6];
      cfa[row + x] *= wb[ch];
    }
  }
}

/**
 * Pad CFA to align with the canonical 6x6 pattern.
 * Uses reflect padding at top and left edges.
 *
 * @param {Float32Array} cfa
 * @param {number} width
 * @param {number} height
 * @param {number} dy
 * @param {number} dx
 * @returns {{ data: Float32Array, width: number, height: number, padTop: number, padLeft: number }}
 */
export function padToAlignment(cfa, width, height, dy, dx) {
  // dy, dx are in additive convention: ref[(y+dy)%6][(x+dx)%6] = vis(y,x)
  // Padding by (dy, dx) rows/cols aligns padded data with reference at (0,0):
  //   padded(Y,X) = vis(Y-dy, X-dx) → CFA = ref[((Y-dy)+dy)%6][((X-dx)+dx)%6] = ref[Y%6][X%6]
  const padTop = dy;
  const padLeft = dx;

  if (padTop === 0 && padLeft === 0) {
    return { data: cfa, width, height, padTop: 0, padLeft: 0 };
  }

  const newW = width + padLeft;
  const newH = height + padTop;
  const out = new Float32Array(newW * newH);

  for (let y = 0; y < newH; y++) {
    // Reflect-pad: mirror the source row index
    const srcY = y < padTop ? padTop - 1 - y : y - padTop;
    const clampedSrcY = Math.min(srcY, height - 1);
    for (let x = 0; x < newW; x++) {
      const srcX = x < padLeft ? padLeft - 1 - x : x - padLeft;
      const clampedSrcX = Math.min(srcX, width - 1);
      out[y * newW + x] = cfa[clampedSrcY * width + clampedSrcX];
    }
  }

  return { data: out, width: newW, height: newH, padTop, padLeft };
}

/**
 * Compute tile grid coordinates and zero-pad the CFA to fit.
 *
 * @param {Float32Array} cfa
 * @param {number} width
 * @param {number} height
 * @param {number} patchSize
 * @param {number} overlap
 * @returns {{ tiles: Array<{x: number, y: number}>, paddedCfa: Float32Array, hPad: number, wPad: number }}
 */
export function generateTiles(cfa, width, height, patchSize, overlap) {
  const stride = patchSize - overlap;
  const hPad = Math.ceil((height - overlap) / stride) * stride + patchSize;
  const wPad = Math.ceil((width - overlap) / stride) * stride + patchSize;

  // Zero-pad
  const paddedCfa = new Float32Array(hPad * wPad);
  for (let y = 0; y < height; y++) {
    paddedCfa.set(cfa.subarray(y * width, y * width + width), y * wPad);
  }

  // Generate tile coordinates
  const tiles = [];
  for (let y = 0; y <= hPad - patchSize; y += stride) {
    for (let x = 0; x <= wPad - patchSize; x += stride) {
      tiles.push({ x, y });
    }
  }

  return { tiles, paddedCfa, hPad, wPad };
}

/**
 * Precompute channel masks for a given patch size.
 * Returns three flat Float32Array masks (R, G, B) of size patchSize^2.
 */
export function makeChannelMasks(patchSize) {
  const n = patchSize * patchSize;
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  for (let y = 0; y < patchSize; y++) {
    const patY = y % 6;
    const row = y * patchSize;
    for (let x = 0; x < patchSize; x++) {
      const ch = XTRANS_PATTERN[patY][x % 6];
      const idx = row + x;
      if (ch === 0) r[idx] = 1;
      else if (ch === 1) g[idx] = 1;
      else b[idx] = 1;
    }
  }
  return { r, g, b };
}

/**
 * Build a single tile's input tensor: [CFA, R_mask, G_mask, B_mask].
 * Returns Float32Array of shape [4, patchSize, patchSize] in row-major order.
 *
 * @param {Float32Array} paddedCfa - Zero-padded full CFA
 * @param {number} paddedWidth - Width of padded CFA
 * @param {number} x - Tile left coordinate
 * @param {number} y - Tile top coordinate
 * @param {number} patchSize
 * @param {{ r: Float32Array, g: Float32Array, b: Float32Array }} masks
 * @returns {Float32Array}
 */
export function buildTileInput(paddedCfa, paddedWidth, x, y, patchSize, masks) {
  const n = patchSize * patchSize;
  const input = new Float32Array(4 * n);

  // Channel 0: CFA values
  for (let py = 0; py < patchSize; py++) {
    const srcOffset = (y + py) * paddedWidth + x;
    const dstOffset = py * patchSize;
    input.set(paddedCfa.subarray(srcOffset, srcOffset + patchSize), dstOffset);
  }

  // Channels 1-3: R, G, B masks (precomputed, same for every tile)
  input.set(masks.r, n);
  input.set(masks.g, 2 * n);
  input.set(masks.b, 3 * n);

  return input;
}
