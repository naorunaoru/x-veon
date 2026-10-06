import type { Host, LibraryHost, LibraryPhoto } from '@/host';
import { BUILD } from '@/lib/channel';
import type { DesktopBridge } from '../protocol/bridge';
import { createWorkerClient } from './port';
import { createExporter } from './exporter';
import { createDisplayHost } from './display';
import { BENCH_SAMPLE } from '@/dev/golden-contract';
const NAMES = new Set(['DSCF3332.RAF', 'sony_a6400_21.arw', BENCH_SAMPLE]);
/** Golden fixtures keep edits and fetched RAW files in memory. */
export function createGoldenHost(bridge: DesktopBridge) {
  const client = createWorkerClient(bridge, () => {});
  const exporter = createExporter(bridge, client);
  const photos = new Map<string, LibraryPhoto>(),
    files = new Map<string, File>();
  const library: LibraryHost = {
    async load() {
      return { photos: [...photos.values()], complete: true };
    },
    async addFiles(incoming) {
      if (incoming.some((file) => !NAMES.has(file.name)))
        throw Error('The golden host accepts only its named sample RAWs.');
      const added = incoming.map((file) => {
        const photo: LibraryPhoto = {
          id: crypto.randomUUID().replaceAll('-', '').slice(0, 22),
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
    host: { library, exporter, display: createDisplayHost(bridge), build: BUILD, settingsDbName: 'xveon-desktop-golden' } satisfies Host,
    releaseFixture(id: string) {
      photos.delete(id);
      files.delete(id);
    },
  };
}
