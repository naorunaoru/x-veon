/**
 * LensFun database client for the browser.
 *
 * Fetches the pre-converted JSON database from <base>/lensfun/ (see lensfunUrl), matches EXIF
 * camera + lens strings against it, and returns calibration profiles
 * (distortion, TCA, vignetting) for GPU lens correction.
 */

import type { LfCamera, LfLens, LensProfile } from '@/lib/lensfun-types';

export type { LfCamera, LfDistortion, LfTCA, LfVignetting, LfLens, LensProfile } from '@/lib/lensfun-types';

// ── Database file shapes (the record types live in lib/lensfun-types) ────

interface LfDbFile {
  cameras: LfCamera[];
  lenses: LfLens[];
}

interface LfIndexEntry {
  file: string;
  cameras: string[];
  lenses: string[];
}

// ── In-memory cache ─────────────────────────────────────────────────

let indexCache: LfIndexEntry[] | null = null;
const dbFileCache = new Map<string, LfDbFile>();

/** URL of a generated lens-data file (public/lensfun, built by `npm run build:lensfun`), relative to the deployed base path. */
export function lensfunUrl(filename: string, base: string = import.meta.env.BASE_URL): string {
  const prefix = base.endsWith('/') ? base : `${base}/`;
  return `${prefix}lensfun/${filename}`;
}

async function fetchJson<T>(filename: string): Promise<T> {
  const url = lensfunUrl(filename);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Lens data missing (${res.status}) at ${url}; run \`npm run build:lensfun\``);
  return (await res.json()) as T;
}

async function fetchIndex(): Promise<LfIndexEntry[]> {
  if (indexCache) return indexCache;
  indexCache = await fetchJson<LfIndexEntry[]>('index.json');
  return indexCache;
}

async function fetchDbFile(filename: string): Promise<LfDbFile> {
  const cached = dbFileCache.get(filename);
  if (cached) return cached;
  const data = await fetchJson<LfDbFile>(filename);
  dbFileCache.set(filename, data);
  return data;
}

// ── String normalization ────────────────────────────────────────────

const MAKE_NOISE = /\b(corporation|co\.?,?ltd\.?|imaging\s*corp\.?|optical\s*co\.?,?ltd\.?|camera\s*ag|digital\s*solutions)\b/gi;

function normalizeMake(s: string): string {
  return s.toLowerCase().replace(MAKE_NOISE, '').replace(/\s+/g, ' ').trim();
}

function normalizeLensStr(s: string): string {
  return s
    .toLowerCase()
    .replace(/[()]/g, '')
    .replace(/\bf\/?\s*/g, '')       // strip "f/" or "f " prefix before aperture
    .replace(/(\d)mm\b/g, '$1')      // strip "mm" suffix from focal lengths
    .replace(/\s+/g, ' ')
    .trim();
}

// ── Third-party lens file prefixes ──────────────────────────────────

const THIRD_PARTY_FILES = [
  'sigma', 'tamron', 'samyang', 'zeiss', 'tokina', 'voigtlander', 'misc',
];

function isThirdPartyFile(filename: string): boolean {
  const lower = filename.toLowerCase();
  return THIRD_PARTY_FILES.some((p) => lower.includes(p));
}

// ── Find relevant DB files for a camera make ────────────────────────

function findRelevantFiles(index: LfIndexEntry[], camera: string): string[] {
  const normCamera = normalizeMake(camera);
  const files = new Set<string>();

  for (const entry of index) {
    for (const cam of entry.cameras) {
      const camMake = normalizeMake(cam.split('|')[0]);
      if (normCamera.startsWith(camMake) || camMake.startsWith(normCamera.split(' ')[0])) {
        files.add(entry.file);
        break;
      }
    }

    // Always include third-party lens manufacturers
    if (isThirdPartyFile(entry.file)) {
      files.add(entry.file);
    }
  }

  return [...files];
}

// ── Camera matching ─────────────────────────────────────────────────

function matchCamera(exifCamera: string, cameras: LfCamera[]): LfCamera | null {
  const norm = normalizeMake(exifCamera);

  // Exact "make model" match
  for (const cam of cameras) {
    if (normalizeMake(`${cam.make} ${cam.model}`) === norm) return cam;
  }

  // Model contained in EXIF string (handles different make spellings)
  const normParts = norm.split(' ');
  for (const cam of cameras) {
    const model = normalizeMake(cam.model);
    const make = normalizeMake(cam.make);
    if (normParts[0].startsWith(make.split(' ')[0]) && norm.includes(model)) {
      return cam;
    }
  }

  return null;
}

// ── Lens scoring ────────────────────────────────────────────────────

function scoreLensMatch(exifLens: string, lens: LfLens, mount: string | null): number {
  const exifNorm = normalizeLensStr(exifLens);
  const candidates = [lens.model, ...lens.aliases];
  let bestStringScore = 0;

  for (const cand of candidates) {
    const candNorm = normalizeLensStr(cand);

    // Exact match
    if (candNorm === exifNorm) {
      bestStringScore = 1.0;
      break;
    }

    // One contains the other
    if (candNorm.includes(exifNorm) || exifNorm.includes(candNorm)) {
      const shorter = Math.min(candNorm.length, exifNorm.length);
      const longer = Math.max(candNorm.length, exifNorm.length);
      bestStringScore = Math.max(bestStringScore, shorter / longer);
      continue;
    }

    // Token overlap (Jaccard similarity)
    const exifTokens = new Set(exifNorm.split(/[\s/\-]+/).filter(Boolean));
    const candTokens = new Set(candNorm.split(/[\s/\-]+/).filter(Boolean));
    let overlap = 0;
    for (const t of exifTokens) if (candTokens.has(t)) overlap++;
    const union = new Set([...exifTokens, ...candTokens]).size;
    const jaccard = union > 0 ? overlap / union : 0;
    bestStringScore = Math.max(bestStringScore, jaccard * 0.9);
  }

  // Mount compatibility bonus
  let mountScore = 0.5;
  if (mount) {
    mountScore = lens.mounts.includes(mount) ? 1.0 : 0.0;
  }

  return bestStringScore * 0.8 + mountScore * 0.2;
}

// ── Profiles ────────────────────────────────────────────────────────

function calibrationCount(lens: LfLens): number {
  return lens.distortion.length + lens.tca.length + lens.vignetting.length;
}

/** The profile of a lens on a camera: the camera's mount and crop factor when it is known, else the lens's own. */
function toProfile(lens: LfLens, camera: LfCamera | null): LensProfile {
  return {
    lensModel: lens.model,
    mount: camera?.mount ?? lens.mounts[0] ?? '',
    cropfactor: camera?.cropfactor ?? lens.cropfactor,
    calibCropfactor: lens.cropfactor,
    calibAspectRatio: lens.aspectRatio ?? 1.5,
    distortion: lens.distortion,
    tca: lens.tca,
    vignetting: lens.vignetting,
  };
}

// ── Public API ──────────────────────────────────────────────────────

export async function matchLens(
  camera: string,
  lensModel: string,
): Promise<LensProfile | null> {
  if (!lensModel) return null;

  const index = await fetchIndex();
  const fileNames = findRelevantFiles(index, camera);
  if (fileNames.length === 0) {
    console.log(`LensFun: no DB files found for camera "${camera}"`);
    return null;
  }

  const dbFiles = await Promise.all(fileNames.map(fetchDbFile));

  // Match camera to find mount + cropfactor
  const allCameras = dbFiles.flatMap((d) => d.cameras);
  const matchedCamera = matchCamera(camera, allCameras);
  const mount = matchedCamera?.mount ?? null;

  if (matchedCamera) {
    console.log(`LensFun: camera "${camera}" → ${matchedCamera.make} ${matchedCamera.model} (mount=${matchedCamera.mount}, crop=${matchedCamera.cropfactor})`);
  } else {
    console.log(`LensFun: camera "${camera}" not found in DB, matching lens without mount filter`);
  }

  // Score all lenses
  const allLenses = dbFiles.flatMap((d) => d.lenses);
  const scored = allLenses
    .map((lens) => ({ lens, score: scoreLensMatch(lensModel, lens, mount) }))
    .filter((s) => s.score >= 0.5)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      // Tiebreak: prefer lens with more calibration data
      return calibrationCount(b.lens) - calibrationCount(a.lens);
    });

  if (scored.length === 0) {
    console.log(`LensFun: no match for lens "${lensModel}" (searched ${allLenses.length} lenses across ${fileNames.length} files)`);
    return null;
  }

  const best = scored[0];
  console.log(
    `LensFun: matched "${lensModel}" → "${best.lens.model}" (score=${best.score.toFixed(2)}, ` +
    `dist=${best.lens.distortion.length}, tca=${best.lens.tca.length}, vig=${best.lens.vignetting.length})`,
  );

  return toProfile(best.lens, matchedCamera);
}

/** A lens to pick by hand: where it lives (`file`) and how the list shows it. */
export interface LensChoice { file: string; make: string; model: string }

const byName = new Intl.Collator('en');

/** The lenses to choose from for a camera: those of its make's files and of the third-party files, from the index alone. */
export async function listLenses(camera: string): Promise<LensChoice[]> {
  const index = await fetchIndex();
  const files = findRelevantFiles(index, camera);
  const choices = new Map<string, LensChoice>();
  for (const entry of index) {
    if (!files.includes(entry.file)) continue;
    for (const name of entry.lenses) {
      // "Make|Model": the model can hold a bar of its own ("… DC DN | Contemporary 016"), the make never does.
      const bar = name.indexOf('|');
      choices.set(`${entry.file}|${name}`, { file: entry.file, make: name.slice(0, bar), model: name.slice(bar + 1) });
    }
  }
  return [...choices.values()].sort((a, b) => byName.compare(a.make, b.make) || byName.compare(a.model, b.model));
}

async function findCamera(camera: string): Promise<LfCamera | null> {
  const index = await fetchIndex();
  const dbFiles = await Promise.all(findRelevantFiles(index, camera).map(fetchDbFile));
  return matchCamera(camera, dbFiles.flatMap((d) => d.cameras));
}

/** The profile of a hand-picked lens on this camera; null when the file has no such model. */
export async function loadLensProfile(
  choice: { file: string; model: string },
  camera: string,
): Promise<LensProfile | null> {
  const [file, matchedCamera] = await Promise.all([fetchDbFile(choice.file), findCamera(camera)]);
  // A file can list one model several times (one entry per mount or crop factor): take the entry that fits the camera.
  const fitsMount = (lens: LfLens) => (matchedCamera && lens.mounts.includes(matchedCamera.mount) ? 1 : 0);
  const [lens] = file.lenses
    .filter((l) => l.model === choice.model)
    .sort((a, b) => fitsMount(b) - fitsMount(a) || calibrationCount(b) - calibrationCount(a));
  return lens ? toProfile(lens, matchedCamera) : null;
}
