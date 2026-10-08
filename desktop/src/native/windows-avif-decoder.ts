import { execFileSync } from 'node:child_process';

// Fixed reference, independent of the encoder under test: eight 64px color patches,
// 512x64, 10-bit full-range BT.2020 YCbCr/HLG, generated at 52a91f4.
// Verified with macOS ImageIO and libavif; SHA-256 is pinned in the helper tests.
export const knownGoodAvif = Buffer.from(
  'AAAAGGZ0eXBhdmlmAAAAAG1pZjFtaWFmAAAA5m1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAHBpY3QAAAAAAAAAAAAAAAAAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABBgAAAHUAAAAjaWluZgAAAAAAAQAAABVpbmZlAgAAAAABAABhdjAxAAAAAGppcHJwAAAAS2lwY28AAAAUaXNwZQAAAAAAAAIAAAAAQAAAAAxhdjFDgT9AAAAAABBwaXhpAAAAAAMKCgoAAAATY29scm5jbHgACQASAAmAAAAAF2lwbWEAAAAAAAAAAQABBAGCAwQAAAB9bWRhdBIACg0gAAD6F//4hCrCRIJ0MmIQAg6BxcRCQbAACAAEAAAAAAAAAAggggAACLr9TraJYaxgTvx6R6amZ5FBPZfZbJKkcWCum101iMunkwuVCZmMVeZejEy+NVrAFsQc4EkF6hWi1dVekuSmdtmqpalB5021WA==', 'base64',
);

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
  $causes = @()
  $cause = $_.Exception
  while ($null -ne $cause) {
    $causes += @{ type=$cause.GetType().FullName; hResult=$cause.HResult; message=$cause.Message }
    $cause = $cause.InnerException
  }
  @{ error=@($causes) } | ConvertTo-Json -Depth 4 -Compress
  exit 0
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

export interface DecoderCause { type: string; hResult: number; message: string }
export type DecoderResult = { centers: number[][] } | { error: DecoderCause[] };
export type Decode = (file: string) => DecoderResult;

function runDecoder(file: string): DecoderResult {
  return JSON.parse(execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', decodeWindows], {
    env: { ...process.env, XV_AVIF_COLOR_PROBE: file }, encoding: 'utf8', timeout: 30_000,
  })) as DecoderResult;
}

function decodeFailure(file: string, causes: DecoderCause[]): Error {
  return new Error(`Windows AVIF decode failed for ${file}: ${JSON.stringify(causes)}`);
}

/** Only the fixed reference may establish codec unavailability. Once it decodes,
 * every generated-file error is fatal, even if WPF reports NotSupportedException. */
export function decodeWindowsAvif(reference: string, file: string, decode: Decode = runDecoder):
  { centers: number[][] } | { unavailable: true; causes: DecoderCause[] } {
  const probe = decode(reference);
  if ('error' in probe) {
    // WPF can translate WIC's COM error into a managed NotSupportedException,
    // losing the original HRESULT. Never classify the candidate image this way.
    if (probe.error.some(cause => cause.hResult === -2003292336 // WINCODEC_ERR_COMPONENTNOTFOUND
      || (cause.type === 'System.NotSupportedException' && cause.hResult === -2146233067))) {
      return { unavailable: true, causes: probe.error };
    }
    throw decodeFailure(reference, probe.error);
  }
  const result = decode(file);
  if ('error' in result) throw decodeFailure(file, result.error);
  return result;
}
