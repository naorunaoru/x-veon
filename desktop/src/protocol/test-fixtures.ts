import type { LibraryPhoto, PhotoFacts } from '@/host';
import { defaultPhotoEdit } from '@/app/photo-edit';
export function realisticFacts(): PhotoFacts {
  return {
    cfaType: 'xtrans', status: 'done', error: null, resultMethod: 'neural-net',
    metadata: { camera: 'Fujifilm X-T5', lensModel: 'XF16-55mmF2.8 R LM WR', focalLength: 35, fNumber: 5.6 },
    resultMeta: {
      exportData: { width: 7728, height: 5152, xyzToCam: [1, 0, 0, 0, 1, 0, 0, 0, 1], wbCoeffs: [2.1, 1, 1.6, 1], camToXyz: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], orientation: 'normal' },
      metadata: { make: 'Fujifilm', model: 'X-T5', width: 7728, height: 5152, tileCount: 120, inferenceTime: 3210, backend: 'webgpu', exposureBias: 0.33, lensModel: 'XF16-55mmF2.8 R LM WR', focalLength: 35, fNumber: 5.6, colorTemp: 5500, tint: 0, modelSize: 'M', modelIdentity: { size: 'M', sha256: 'a'.repeat(64) }, cfaType: 'xtrans', modelNote: null },
    },
    lensProfile: { lensModel: 'XF16-55mmF2.8 R LM WR', mount: 'Fujifilm X', cropfactor: 1.5, distortion: [{ model: 'ptlens', focal: 35, a: 0.01, b: -0.02, c: 0.003 }], tca: [{ model: 'poly3', focal: 35, vr: 1, vb: 1 }], vignetting: [{ model: 'pa', focal: 35, aperture: 5.6, distance: 10, k1: -0.4, k2: 0.1, k3: 0.02 }] },
  };
}
export function photo(index = 0): LibraryPhoto {
  return { id: String(index).padStart(22, '0'), name: `DSCF${index}`, originalName: `DSCF${index}.RAF`, fileSize: 84234912, thumbnailUrl: `xveon-photo://thumb/${String(index).padStart(22, '0')}`, edit: defaultPhotoEdit(), facts: realisticFacts(), editing: 'saved', editingNote: null };
}
