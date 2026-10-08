import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { expect, it } from 'vitest';
import { loadNativeEncoder } from '../worker/native';

// Decode through the installed Windows codec, not Chromium/libavif: they can
// correctly read the AV1 payload even when Windows misinterprets the container.
const decodeWindows = String.raw`
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName PresentationCore
try {
  $decoder = [System.Windows.Media.Imaging.BitmapDecoder]::Create(
    [uri]$env:XV_AVIF_COLOR_PROBE,
    [System.Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat,
    [System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad)
} catch {
  $cause = $_.Exception
  while ($null -ne $cause) {
    if ($cause.HResult -eq -2003292336) { # WINCODEC_ERR_COMPONENTNOTFOUND
      '{"unavailable":true}'
      exit 0
    }
    $cause = $cause.InnerException
  }
  throw
}
# Keep HDR values unclipped; converting to 8-bit would turn bright grays white.
$image = [System.Windows.Media.Imaging.FormatConvertedBitmap]::new(
  $decoder.Frames[0], [System.Windows.Media.PixelFormats]::Rgba128Float, $null, 0)
$pixels = [byte[]]::new($image.PixelWidth * $image.PixelHeight * 16)
$image.CopyPixels($pixels, $image.PixelWidth * 16, 0)
$centers = @()
for ($i = 0; $i -lt 8; $i++) {
  $offset = (32 * $image.PixelWidth + $i * 64 + 32) * 16
  $centers += ,@([BitConverter]::ToSingle($pixels, $offset),
    [BitConverter]::ToSingle($pixels, $offset + 4),
    [BitConverter]::ToSingle($pixels, $offset + 8))
}
@{centers=$centers} | ConvertTo-Json -Depth 3 -Compress
`;

it.skipIf(process.platform !== 'win32')('Windows AVIF decoder preserves neutral grays and primary colors', async ({ skip }) => {
  const loaded = loadNativeEncoder(createRequire(import.meta.url), path.resolve(__dirname, '../../native'));
  if (!loaded.ok) throw new Error(loaded.reason);
  const patches = [[0, 0, 0], [.01, .01, .01], [.18, .18, .18], [.5, .5, .5], [1, 1, 1], [.18, 0, 0], [0, .18, 0], [0, 0, .18]];
  const width = patches.length * 64, height = 64;
  const data = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) data.set(patches[Math.floor(x / 64)], (y * width + x) * 3);
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-avif-color-'));
  try {
    const file = path.join(dir, 'patches.avif');
    await fs.writeFile(file, await loaded.encoder.encode(data, null, {
      format: 'avif', width, height, orientation: 'Normal', quality: 100, peakLuminance: 1000, threads: 1,
    }));
    const result = JSON.parse(execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', decodeWindows], {
      env: { ...process.env, XV_AVIF_COLOR_PROBE: file }, encoding: 'utf8', timeout: 30_000,
    })) as { unavailable?: boolean; centers: number[][] };
    // Windows CI images may lack the optional HEIF codec. Other decoder errors fail.
    if (result.unavailable) { skip(); return; }
    expect(result.centers).toHaveLength(8);
    const grays = result.centers.slice(0, 5);
    expect(grays[0].every(v => Math.abs(v) < .001)).toBe(true);
    for (const rgb of grays.slice(1)) {
      expect(rgb.every(Number.isFinite)).toBe(true);
      expect(Math.max(...rgb) - Math.min(...rgb)).toBeLessThan(Math.max(...rgb) * .02);
    }
    for (let i = 1; i < grays.length; i++) expect(grays[i][0]).toBeGreaterThan(grays[i - 1][0]);
    for (let channel = 0; channel < 3; channel++) {
      const rgb = result.centers[5 + channel];
      expect(rgb[channel]).toBeGreaterThan(0);
      for (let other = 0; other < 3; other++) if (other !== channel) expect(rgb[other]).toBeLessThan(rgb[channel] * .1);
    }
  } finally { await fs.rm(dir, { recursive: true, force: true }); }
});
