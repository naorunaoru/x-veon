/**
 * LensFun database client for the browser.
 *
 * Fetches the pre-converted JSON database from /lensfun/, matches EXIF
 * camera + lens strings against it, and returns calibration profiles
 * (distortion, TCA, vignetting) for GPU lens correction.
 */

// ── LensFun data model (matches JSON from convert-lensfun-db.ts) ────

export interface LfCamera {
  make: string;
  model: string;
  mount: string;
  cropfactor: number;
  variant?: string;
}

export interface LfDistortion {
  model: string;
  focal: number;
  k1?: number;
  k2?: number;
  a?: number;
  b?: number;
  c?: number;
}

export interface LfTCA {
  model: string;
  focal: number;
  vr?: number;
  vb?: number;
  br?: number;
  cr?: number;
  bb?: number;
  cb?: number;
  kr?: number;
  kb?: number;
}

export interface LfVignetting {
  model: string;
  focal: number;
  aperture: number;
  distance: number;
  k1: number;
  k2: number;
  k3: number;
}

export interface LfLens {
  make: string;
  model: string;
  aliases: string[];
  mounts: string[];
  cropfactor: number;
  type?: string;
  distortion: LfDistortion[];
  tca: LfTCA[];
  vignetting: LfVignetting[];
}

interface LfDbFile {
  cameras: LfCamera[];
  lenses: LfLens[];
}

interface LfIndexEntry {
  file: string;
  cameras: string[];
  lenses: string[];
}

// ── Match result (stored on QueuedFile) ─────────────────────────────

export interface LensProfile {
  lensModel: string;
  mount: string;
  cropfactor: number;
  distortion: LfDistortion[];
  tca: LfTCA[];
  vignetting: LfVignetting[];
}

// ── In-memory cache ─────────────────────────────────────────────────

let indexCache: LfIndexEntry[] | null = null;
const dbFileCache = new Map<string, LfDbFile>();

async function fetchIndex(): Promise<LfIndexEntry[]> {
  if (indexCache) return indexCache;
  const res = await fetch('/lensfun/index.json');
  indexCache = await res.json();
  return indexCache!;
}

async function fetchDbFile(filename: string): Promise<LfDbFile> {
  const cached = dbFileCache.get(filename);
  if (cached) return cached;
  const res = await fetch(`/lensfun/${filename}`);
  const data: LfDbFile = await res.json();
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
      const aCount = a.lens.distortion.length + a.lens.tca.length + a.lens.vignetting.length;
      const bCount = b.lens.distortion.length + b.lens.tca.length + b.lens.vignetting.length;
      return bCount - aCount;
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

  return {
    lensModel: best.lens.model,
    mount: mount ?? best.lens.mounts[0] ?? '',
    cropfactor: matchedCamera?.cropfactor ?? best.lens.cropfactor,
    distortion: best.lens.distortion,
    tca: best.lens.tca,
    vignetting: best.lens.vignetting,
  };
}
