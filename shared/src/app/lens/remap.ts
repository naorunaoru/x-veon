import type { QueuedFile } from '@/app/store';
import { lensPickKey } from '@/app/store/lens';
import { effectiveLensEdit } from '@/lib/lens/edit';
import { parseFujiTables, type FujiKnots } from '@/lib/lens/fuji-tables';
import type { LensProfile } from '@/lib/lensfun-types';
import type { ProcessingResultMeta } from '@/lib/types';
import { buildLensTable, type LensFrame, type LensRemap } from '@/renderer/lens-table';
import { resolveLens, type ResolvedLens } from './resolve';

/** What a photo's decode tells the lens correction: the camera's tables and the texture they apply to. */
interface Decoded { result: ProcessingResultMeta; camera: FujiKnots | null; frame: LensFrame }

function decodedOf(file: QueuedFile): Decoded | null {
  const { result } = file;
  if (!result) return null;
  const fuji = result.metadata.fujiCorrection;
  const camera = fuji ? parseFujiTables(fuji) : null;
  const frame: LensFrame = { width: result.exportData.width, height: result.exportData.height };
  if (fuji?.centre) frame.centre = fuji.centre;
  if (camera && camera.halfDiagPx > 0) frame.halfDiagPx = camera.halfDiagPx;
  return { result, camera, frame };
}

function resolveDecoded(file: QueuedFile, picks: Record<string, LensProfile>, { result, camera, frame }: Decoded): ResolvedLens {
  const edit = effectiveLensEdit(file.edit);
  // The result's focal length and f-number, else the quick metadata's; 0 means unknown.
  const known = (x: number, quick: number | undefined) => (x > 0 ? x : quick ?? 0);
  // undefined while a hand-picked profile is not loaded
  const picked = typeof edit.source === 'object'
    ? (picks[lensPickKey(edit.source.lensfun.file, edit.source.lensfun.model)] as LensProfile | undefined)
    : null;
  return resolveLens({
    edit,
    camera,
    lensfun: file.lensProfile,
    picked,
    focal: known(result.metadata.focalLength, file.metadata?.focalLength),
    fNumber: known(result.metadata.fNumber, file.metadata?.fNumber),
    halfDiagPx: frame.halfDiagPx ?? Math.hypot(frame.width, frame.height) / 2,
  });
}

/** The decoded texture, with the camera's optical centre and half-diagonal where the photo carries them. Null before a result exists. */
export function lensFrameFor(file: QueuedFile): LensFrame | null {
  return decodedOf(file)?.frame ?? null;
}

/** Which source provides each correction for this photo, and the curves. Null before a result exists. */
export function resolvedLensFor(file: QueuedFile, picks: Record<string, LensProfile>): ResolvedLens | null {
  const decoded = decodedOf(file);
  return decoded && resolveDecoded(file, picks, decoded);
}

/** The remap the renderer applies to this photo; null when nothing is to be corrected. The canvas and the export both ask for it here. */
export function lensRemapForFile(file: QueuedFile, picks: Record<string, LensProfile>): LensRemap | null {
  const decoded = decodedOf(file);
  if (!decoded) return null;
  const { distortion, ca, vignetting } = effectiveLensEdit(file.edit);
  return buildLensTable(resolveDecoded(file, picks, decoded).curves, { distortion, ca, vignetting }, decoded.frame);
}
