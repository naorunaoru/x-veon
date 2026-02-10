/**
 * WebCodecs-based AVIF encoder.
 *
 * Uses the browser's native AV1 VideoEncoder (potentially hardware-accelerated)
 * to encode a single frame from the HDR canvas, then wraps the AV1 bitstream
 * in a minimal AVIF (ISOBMFF) container with correct CICP color tags.
 */

// --- ISOBMFF helpers ---

function u8(v) { return new Uint8Array([v]); }
function u16be(v) { return new Uint8Array([v >>> 8, v & 0xff]); }
function u32be(v) {
  return new Uint8Array([(v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff]);
}
const ascii = (s) => new TextEncoder().encode(s);

function concat(...arrays) {
  const parts = arrays.map(a => a instanceof ArrayBuffer ? new Uint8Array(a) : a);
  const total = parts.reduce((s, a) => s + a.byteLength, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const a of parts) { result.set(a, offset); offset += a.byteLength; }
  return result;
}

function box(type, payload) {
  return concat(u32be(8 + payload.byteLength), ascii(type), payload);
}

function fullBox(type, version, flags, payload) {
  return box(type, concat(
    new Uint8Array([version, (flags >> 16) & 0xff, (flags >> 8) & 0xff, flags & 0xff]),
    payload,
  ));
}

// --- CICP mapping from VideoColorSpace strings ---

const PRIMARIES_MAP = { 'bt709': 1, 'bt2020': 9, 'smpte432': 12 };
const TRANSFER_MAP = { 'bt709': 1, 'iec61966-2-1': 13, 'smpte2084': 16, 'arib-std-b67': 18 };
const MATRIX_MAP = { 'rgb': 0, 'bt709': 1, 'bt2020-ncl': 9 };

// --- AVIF container builder ---

/**
 * Build a minimal single-image AVIF file from raw AV1 data.
 *
 * @param {ArrayBuffer} av1Data - Raw AV1 bitstream (single key frame)
 * @param {ArrayBuffer} av1cData - AV1CodecConfigurationRecord (from decoderConfig.description)
 * @param {number} width
 * @param {number} height
 * @param {{primaries: number, transfer: number, matrix: number, fullRange: boolean}} cicp
 * @returns {Uint8Array}
 */
function buildAvifContainer(av1Data, av1cData, width, height, cicp) {
  const av1Bytes = new Uint8Array(av1Data);
  const av1cBytes = new Uint8Array(av1cData);

  const ftyp = box('ftyp', concat(ascii('avif'), u32be(0), ascii('avif'), ascii('mif1')));

  const hdlr = fullBox('hdlr', 0, 0, concat(
    u32be(0), ascii('pict'), new Uint8Array(12), u8(0),
  ));

  const pitm = fullBox('pitm', 0, 0, u16be(1));

  const ispe = fullBox('ispe', 0, 0, concat(u32be(width), u32be(height)));
  const av1C = box('av1C', av1cBytes);
  const colr = box('colr', concat(
    ascii('nclx'),
    u16be(cicp.primaries),
    u16be(cicp.transfer),
    u16be(cicp.matrix),
    u8(cicp.fullRange ? 0x80 : 0),
  ));
  const ipco = box('ipco', concat(ispe, av1C, colr));

  const ipma = fullBox('ipma', 0, 0, concat(
    u32be(1), u16be(1), new Uint8Array([3, 0x81, 0x82, 0x83]),
  ));
  const iprp = box('iprp', concat(ipco, ipma));

  // iloc with placeholder extent_offset (patched below)
  const iloc = fullBox('iloc', 0, 0, concat(
    new Uint8Array([0x44, 0x00]),  // offset_size=4, length_size=4, base_offset_size=0
    u16be(1), u16be(1), u16be(0), u16be(1), // item_count, item_ID, data_ref, extent_count
    u32be(0),                       // extent_offset (PLACEHOLDER)
    u32be(av1Bytes.byteLength),     // extent_length
  ));

  const meta = fullBox('meta', 0, 0, concat(hdlr, pitm, iloc, iprp));
  const mdat = box('mdat', av1Bytes);

  const file = concat(ftyp, meta, mdat);

  // Patch iloc extent_offset to point at mdat payload
  const mdatPayloadOffset = ftyp.byteLength + meta.byteLength + 8; // +8 = mdat box header
  // iloc position: meta starts after ftyp, skip meta's fullBox header (12),
  // then hdlr, pitm to reach iloc, then iloc's fullBox header (12) + 10 bytes to extent_offset
  const ilocStart = ftyp.byteLength + 12 + hdlr.byteLength + pitm.byteLength;
  const extentOffsetPos = ilocStart + 12 + 10;
  new DataView(file.buffer).setUint32(extentOffsetPos, mdatPayloadOffset);

  return file;
}

// --- WebCodecs AV1 encoding ---

/**
 * AV1 codec string based on image dimensions.
 * Format: av01.P.LLT.DD (profile.level+tier.bitdepth)
 */
function av1Codec(width, height) {
  const pixels = width * height;
  // AV1 levels: 8=4.0 (≤2M px), 12=5.0 (≤8.9M px), 16=6.0 (≤35.6M px)
  const level = pixels <= 2228224 ? 8 : pixels <= 8912896 ? 12 : 16;
  return `av01.0.${String(level).padStart(2, '0')}M.10`;
}

/**
 * Check if WebCodecs AV1 encoding is supported for the given dimensions.
 */
export async function isWebCodecsAvifSupported(width, height) {
  if (typeof VideoEncoder === 'undefined') return false;
  try {
    const support = await VideoEncoder.isConfigSupported({
      codec: av1Codec(width || 1920, height || 1080),
      width: width || 1920,
      height: height || 1080,
      bitrate: 10_000_000,
    });
    return support.supported === true;
  } catch {
    return false;
  }
}

/**
 * Encode the HDR canvas as AVIF using WebCodecs VideoEncoder.
 *
 * @param {HTMLCanvasElement} canvas - rec2100-hlg canvas with rendered image
 * @param {number} quality - 1-100
 * @returns {Promise<Blob>} AVIF file blob
 */
export async function encodeAvifWebCodecs(canvas, quality) {
  const { width, height } = canvas;
  const codec = av1Codec(width, height);

  // Quality → bitrate mapping (with framerate=1, all budget goes to our single frame)
  // Exponential: quality 1 → ~500kbps, quality 50 → ~7Mbps, quality 100 → ~100Mbps
  const bitrate = Math.round(500_000 * Math.pow(200, quality / 100));

  return new Promise((resolve, reject) => {
    let av1Data = null;
    let av1cData = null;
    let colorSpace = null;

    const encoder = new VideoEncoder({
      output: (chunk, metadata) => {
        const buf = new ArrayBuffer(chunk.byteLength);
        chunk.copyTo(buf);
        av1Data = buf;

        if (metadata?.decoderConfig?.description) {
          av1cData = metadata.decoderConfig.description;
        }
        if (metadata?.decoderConfig?.colorSpace) {
          colorSpace = metadata.decoderConfig.colorSpace;
        }
      },
      error: (e) => reject(e),
    });

    encoder.configure({
      codec,
      width,
      height,
      bitrate,
      framerate: 1,
    });

    const frame = new VideoFrame(canvas, { timestamp: 0 });
    encoder.encode(frame, { keyFrame: true });
    frame.close();

    encoder.flush().then(() => {
      encoder.close();

      if (!av1Data) {
        reject(new Error('WebCodecs AV1 encoding produced no output'));
        return;
      }

      // Read CICP from encoder's color space, fall back to BT.2020/HLG defaults
      const cicp = {
        primaries: PRIMARIES_MAP[colorSpace?.primaries] ?? 9,
        transfer: TRANSFER_MAP[colorSpace?.transfer] ?? 18,
        matrix: MATRIX_MAP[colorSpace?.matrix] ?? 9,
        fullRange: colorSpace?.fullRange ?? true,
      };

      // If no av1C from decoderConfig, we can't build a valid AVIF
      if (!av1cData) {
        reject(new Error('WebCodecs did not provide AV1 codec configuration'));
        return;
      }

      console.log(`WebCodecs AVIF: ${width}x${height}, ${(av1Data.byteLength / 1024).toFixed(0)}KB, ` +
        `CICP(${cicp.primaries}/${cicp.transfer}/${cicp.matrix})`);

      const avif = buildAvifContainer(av1Data, av1cData, width, height, cicp);
      resolve(new Blob([avif], { type: 'image/avif' }));
    }).catch(reject);
  });
}
