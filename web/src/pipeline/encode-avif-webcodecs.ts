// ISOBMFF helpers

function u8(v: number): Uint8Array { return new Uint8Array([v]); }
function u16be(v: number): Uint8Array { return new Uint8Array([v >>> 8, v & 0xff]); }
function u32be(v: number): Uint8Array {
  return new Uint8Array([(v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff]);
}
const ascii = (s: string) => new TextEncoder().encode(s);

function concat(...arrays: (Uint8Array | ArrayBuffer)[]): Uint8Array {
  const parts = arrays.map(a => a instanceof ArrayBuffer ? new Uint8Array(a) : a);
  const total = parts.reduce((s, a) => s + a.byteLength, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const a of parts) { result.set(a, offset); offset += a.byteLength; }
  return result;
}

function box(type: string, payload: Uint8Array): Uint8Array {
  return concat(u32be(8 + payload.byteLength), ascii(type), payload);
}

function fullBox(type: string, version: number, flags: number, payload: Uint8Array): Uint8Array {
  return box(type, concat(
    new Uint8Array([version, (flags >> 16) & 0xff, (flags >> 8) & 0xff, flags & 0xff]),
    payload,
  ));
}

// CICP mapping from VideoColorSpace strings

const PRIMARIES_MAP: Record<string, number> = { 'bt709': 1, 'bt2020': 9, 'smpte432': 12 };
const TRANSFER_MAP: Record<string, number> = { 'bt709': 1, 'iec61966-2-1': 13, 'smpte2084': 16, 'arib-std-b67': 18 };
const MATRIX_MAP: Record<string, number> = { 'rgb': 0, 'bt709': 1, 'bt2020-ncl': 9 };

interface CicpInfo {
  primaries: number;
  transfer: number;
  matrix: number;
  fullRange: boolean;
}

function buildAvifContainer(
  av1Data: ArrayBuffer, av1cData: ArrayBuffer,
  width: number, height: number, cicp: CicpInfo,
): Uint8Array {
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

  const iloc = fullBox('iloc', 0, 0, concat(
    new Uint8Array([0x44, 0x00]),
    u16be(1), u16be(1), u16be(0), u16be(1),
    u32be(0),
    u32be(av1Bytes.byteLength),
  ));

  const meta = fullBox('meta', 0, 0, concat(hdlr, pitm, iloc, iprp));
  const mdat = box('mdat', av1Bytes);

  const file = concat(ftyp, meta, mdat);

  const mdatPayloadOffset = ftyp.byteLength + meta.byteLength + 8;
  const ilocStart = ftyp.byteLength + 12 + hdlr.byteLength + pitm.byteLength;
  const extentOffsetPos = ilocStart + 12 + 10;
  new DataView(file.buffer).setUint32(extentOffsetPos, mdatPayloadOffset);

  return file;
}

function av1Codec(width: number, height: number): string {
  const pixels = width * height;
  const level = pixels <= 2228224 ? 8 : pixels <= 8912896 ? 12 : 16;
  return `av01.0.${String(level).padStart(2, '0')}M.10`;
}

export async function isWebCodecsAvifSupported(width: number, height: number): Promise<boolean> {
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

export async function encodeAvifWebCodecs(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  const { width, height } = canvas;
  const codec = av1Codec(width, height);

  const bitrate = Math.round(500_000 * Math.pow(200, quality / 100));

  return new Promise((resolve, reject) => {
    let av1Data: ArrayBuffer | null = null;
    let av1cData: ArrayBuffer | null = null;
    let colorSpace: VideoColorSpaceInit | null = null;

    const encoder = new VideoEncoder({
      output: (chunk, metadata) => {
        const buf = new ArrayBuffer(chunk.byteLength);
        chunk.copyTo(buf);
        av1Data = buf;

        if (metadata?.decoderConfig?.description) {
          av1cData = metadata.decoderConfig.description as ArrayBuffer;
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

      const cicp: CicpInfo = {
        primaries: PRIMARIES_MAP[colorSpace?.primaries ?? ''] ?? 9,
        transfer: TRANSFER_MAP[colorSpace?.transfer ?? ''] ?? 18,
        matrix: MATRIX_MAP[colorSpace?.matrix ?? ''] ?? 9,
        fullRange: colorSpace?.fullRange ?? true,
      };

      if (!av1cData) {
        reject(new Error('WebCodecs did not provide AV1 codec configuration'));
        return;
      }

      console.log(`WebCodecs AVIF: ${width}x${height}, ${(av1Data!.byteLength / 1024).toFixed(0)}KB, ` +
        `CICP(${cicp.primaries}/${cicp.transfer}/${cicp.matrix})`);

      const avif = buildAvifContainer(av1Data!, av1cData!, width, height, cicp);
      resolve(new Blob([avif.buffer as ArrayBuffer], { type: 'image/avif' }));
    }).catch(reject);
  });
}
