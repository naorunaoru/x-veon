import type { Host, LibraryHost, LibraryPhoto } from '@/host';
import { BUILD } from '@/lib/channel';
import { createExporter } from './exporter';
import { createDisplayHost } from './display';
const NAMES = new Set(['DSCF3332.RAF', 'sony_a6400_21.arw']);
/** Golden fixtures keep edits and fetched RAW files in memory. */
export function createGoldenHost() {
  const photos = new Map<string, LibraryPhoto>(),
    files = new Map<string, File>();
  const library: LibraryHost = {
    async load() {
      return { photos: [...photos.values()], complete: true };
    },
    async addFiles(incoming) {
      if (incoming.some((file) => !NAMES.has(file.name)))
        throw Error('The golden host accepts only its two sample RAWs.');
      const added = incoming.map((file) => {
        const photo: LibraryPhoto = {
          id: crypto.randomUUID(),
          name: file.name,
          originalName: file.name,
          fileSize: file.size,
          thumbnailUrl: null,
          editing: 'session',
          editingNote: 'Golden: edits are kept only for this session.',
          edit: {
            version: 1,
            lookPreset: 'default',
            openDrtOverrides: {},
            preProcessOverrides: {},
            demosaicMethod: null,
            model: null,
          },
          facts: {
            cfaType: file.name.endsWith('.RAF') ? 'xtrans' : 'bayer',
            metadata: null,
            resultMeta: null,
            resultMethod: null,
            lensProfile: null,
            status: 'queued',
            error: null,
          },
        };
        photos.set(photo.id, photo);
        files.set(photo.id, file);
        return photo;
      });
      return {
        photos: added,
        complete: true,
        selectedIds: added.map((p) => p.id),
      };
    },
    async readRaw(id) {
      const file = files.get(id);
      if (!file) throw Error('Unknown fixture');
      return file.arrayBuffer();
    },
    async saveFacts(id, facts) {
      const p = photos.get(id);
      if (!p) throw Error('Unknown fixture');
      photos.set(id, { ...p, facts });
    },
    async save(id, edit, facts) {
      const p = photos.get(id);
      if (!p) throw Error('Unknown fixture');
      photos.set(id, { ...p, edit, facts });
    },
  };
  return {
    host: { library, exporter: createExporter(), display: createDisplayHost(), build: BUILD, settingsDbName: 'xveon-desktop-golden' } satisfies Host,
    releaseFixture(id: string) {
      photos.delete(id);
      files.delete(id);
    },
  };
}
