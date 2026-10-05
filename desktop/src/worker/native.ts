import path from 'node:path';
import type { ExportFormat } from '@/lib/types';
export interface NativeEncodeInput { format: ExportFormat; width: number; height: number; orientation: string; quality: number; peakLuminance: number; threads?: number }
export interface NativeEncoder { encode(data: Float32Array, hdr: Float32Array | null, input: NativeEncodeInput): Promise<Uint8Array> }
export type NativeLoad = { ok: true; encoder: NativeEncoder } | { ok: false; reason: string };
export function addonFile(platform: NodeJS.Platform = process.platform, arch: string = process.arch): string {
  return `xveon-native.${platform === 'win32' ? `win32-${arch}-msvc` : `${platform}-${arch}`}.node`;
}
/** out/main/worker.js resolves to the app's native directory; Electron redirects unpacked asar files. */
export function loadNativeEncoder(load: (file: string) => unknown = file => require(file), dir = path.resolve(__dirname, '../../native')): NativeLoad {
  const file = path.join(dir, addonFile());
  try {
    const addon = load(file) as { encode?: (d: Float32Array, h: Float32Array | undefined, i: NativeEncodeInput) => Promise<Uint8Array> } | null;
    if (typeof addon?.encode !== 'function') return { ok: false, reason: `The native encoder at ${file} has no encode function.` };
    const encode = addon.encode;
    return { ok: true, encoder: { encode: (data, hdr, input) => encode(data, hdr ?? undefined, input) } };
  } catch (error) {
    return { ok: false, reason: `The native encoder couldn't be loaded: ${error instanceof Error ? error.message : String(error)}` };
  }
}
