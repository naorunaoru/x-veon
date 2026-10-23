import { afterEach, describe, expect, it, vi } from 'vitest';
import type { LfCamera, LfLens } from '@/lib/lensfun-types';

const distortion = [{ model: 'ptlens', focal: 35, a: 0.01, b: -0.02, c: 0.003 }];
const lens = (patch: Partial<LfLens> & Pick<LfLens, 'make' | 'model'>): LfLens => ({
  aliases: [],
  mounts: ['Fujifilm X'],
  cropfactor: 1.5,
  distortion,
  tca: [],
  vignetting: [],
  ...patch,
});
const xt3: LfCamera = { make: 'Fujifilm', model: 'X-T3', mount: 'Fujifilm X', cropfactor: 1.5 };

const index = [
  { file: 'mil-fujifilm.json', cameras: ['Fujifilm|X-T3'], lenses: ['Fujifilm|XF35mmF2 R WR'] },
  { file: 'misc.json', cameras: [], lenses: ['Viltrox|23mmF1.4XM', 'Sigma|Sigma 30mm f/1.4 DC DN | Contemporary 016'] },
  { file: 'mil-canon.json', cameras: ['Canon|EOS R'], lenses: ['Canon|RF 50mm F1.8 STM'] },
];
const bodies = {
  'mil-fujifilm.json': { cameras: [xt3], lenses: [lens({ make: 'Fujifilm', model: 'XF35mmF2 R WR' })] },
  'misc.json': {
    cameras: [],
    lenses: [
      lens({ make: 'Viltrox', model: '23mmF1.4XM' }),
      lens({
        make: 'Sigma', model: 'Sigma 30mm f/1.4 DC DN | Contemporary 016',
        mounts: ['Micro 4/3 System'], cropfactor: 2, aspectRatio: 4 / 3,
      }),
    ],
  },
};

/** Serves the given lens-data files by name; any other file is a 404. */
function stubLensfun(files: Record<string, unknown>) {
  const fetchMock = vi.fn(async (url: string) => {
    const name = url.slice(url.lastIndexOf('/') + 1);
    return name in files ? { ok: true, json: async () => files[name] } : { ok: false, status: 404 };
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}
/** A fresh module instance, so the index and file caches of one test never reach another. */
async function load() {
  vi.resetModules();
  return import('./lensfun');
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('listLenses', () => {
  it("lists the camera's lenses and the third-party files, sorted by make then model", async () => {
    stubLensfun({ 'index.json': index });
    const { listLenses } = await load();
    expect(await listLenses('Fujifilm X-T3')).toEqual([
      { file: 'mil-fujifilm.json', make: 'Fujifilm', model: 'XF35mmF2 R WR' },
      { file: 'misc.json', make: 'Sigma', model: 'Sigma 30mm f/1.4 DC DN | Contemporary 016' },
      { file: 'misc.json', make: 'Viltrox', model: '23mmF1.4XM' },
    ]);
  });

  it('reads the index only, never a database file', async () => {
    const fetchMock = stubLensfun({ 'index.json': index });
    const { listLenses } = await load();
    await listLenses('Fujifilm X-T3');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toMatch(/lensfun\/index\.json$/);
  });

  it('lists only the third-party files for a camera the database does not know', async () => {
    stubLensfun({ 'index.json': index });
    const { listLenses } = await load();
    expect((await listLenses('Acme Z-1')).map((l) => l.file)).toEqual(['misc.json', 'misc.json']);
  });

  it('lists a model once per file, and once in each file that has it', async () => {
    stubLensfun({
      'index.json': [
        { file: 'mil-fujifilm.json', cameras: ['Fujifilm|X-T3'], lenses: ['Fujifilm|XF35mmF2 R WR', 'Fujifilm|XF35mmF2 R WR'] },
        { file: 'misc.json', cameras: [], lenses: ['Fujifilm|XF35mmF2 R WR'] },
      ],
    });
    const { listLenses } = await load();
    expect((await listLenses('Fujifilm X-T3')).map((l) => l.file)).toEqual(['mil-fujifilm.json', 'misc.json']);
  });

  it('returns nothing when the index has no file for the camera', async () => {
    stubLensfun({ 'index.json': [{ file: 'mil-canon.json', cameras: ['Canon|EOS R'], lenses: ['Canon|RF 50mm F1.8 STM'] }] });
    const { listLenses } = await load();
    expect(await listLenses('Fujifilm X-T3')).toEqual([]);
  });
});

describe('loadLensProfile', () => {
  it('loads a hand-picked lens with the camera crop factor and the calibration fields', async () => {
    stubLensfun({ 'index.json': index, ...bodies });
    const { loadLensProfile } = await load();
    const p = await loadLensProfile({ file: 'mil-fujifilm.json', model: 'XF35mmF2 R WR' }, 'Fujifilm X-T3');
    expect(p).toMatchObject({ lensModel: 'XF35mmF2 R WR', cropfactor: 1.5, calibCropfactor: 1.5, calibAspectRatio: 1.5 });
    expect(await loadLensProfile({ file: 'mil-fujifilm.json', model: 'nope' }, 'Fujifilm X-T3')).toBeNull();
  });

  it("keeps the camera's crop factor and mount apart from the lens calibration", async () => {
    stubLensfun({ 'index.json': index, ...bodies });
    const { loadLensProfile } = await load();
    const p = await loadLensProfile(
      { file: 'misc.json', model: 'Sigma 30mm f/1.4 DC DN | Contemporary 016' },
      'Fujifilm X-T3',
    );
    expect(p).toMatchObject({ mount: 'Fujifilm X', cropfactor: 1.5, calibCropfactor: 2, calibAspectRatio: 4 / 3, distortion });
  });

  it('falls back to the lens crop factor and mount for a camera the database does not know', async () => {
    stubLensfun({ 'index.json': index, ...bodies });
    const { loadLensProfile } = await load();
    const p = await loadLensProfile({ file: 'misc.json', model: '23mmF1.4XM' }, 'Acme Z-1');
    expect(p).toMatchObject({ mount: 'Fujifilm X', cropfactor: 1.5, calibCropfactor: 1.5 });
  });

  it('prefers the entry that fits the camera mount when a file lists a model twice', async () => {
    const twice = {
      cameras: [],
      lenses: [
        lens({ make: 'Yongnuo', model: 'YN 35mm f/2', mounts: ['Canon EF'], cropfactor: 1 }),
        lens({ make: 'Yongnuo', model: 'YN 35mm f/2', mounts: ['Fujifilm X'], cropfactor: 1.5 }),
      ],
    };
    stubLensfun({ 'index.json': index, 'mil-fujifilm.json': bodies['mil-fujifilm.json'], 'misc.json': twice });
    const { loadLensProfile } = await load();
    const p = await loadLensProfile({ file: 'misc.json', model: 'YN 35mm f/2' }, 'Fujifilm X-T3');
    expect(p).toMatchObject({ mount: 'Fujifilm X', calibCropfactor: 1.5 });
  });
});

describe('matchLens', () => {
  it('returns the calibration crop factor and aspect ratio next to the image crop factor', async () => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    stubLensfun({ 'index.json': index, ...bodies });
    const { matchLens } = await load();
    const p = await matchLens('Fujifilm X-T3', 'Sigma 30mm f/1.4 DC DN | Contemporary 016');
    expect(p).toMatchObject({ cropfactor: 1.5, calibCropfactor: 2, calibAspectRatio: 4 / 3 });
  });
});
