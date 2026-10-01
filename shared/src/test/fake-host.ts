import { vi } from 'vitest';
import type { Host, LibraryPhoto, PhotoEdit, LibraryHost } from '@/host';
export function defaultEdit(): PhotoEdit {
  return {
    version: 1,
    lookPreset: 'default',
    openDrtOverrides: {},
    preProcessOverrides: {},
    demosaicMethod: null,
    model: null,
  };
}
export function fakePhoto(id = 'a'): LibraryPhoto {
  return {
    id,
    name: id,
    originalName: `${id}.raf`,
    fileSize: 42,
    thumbnailUrl: null,
    edit: defaultEdit(),
    editing: 'saved',
    editingNote: null,
    facts: {
      cfaType: 'xtrans',
      metadata: null,
      resultMeta: null,
      resultMethod: null,
      lensProfile: null,
      status: 'queued',
      error: null,
    },
  };
}
export function fakeHost(library: Partial<LibraryHost> = {}): Host {
  return {
    library: {
      load: vi.fn(async () => ({ photos: [], complete: true })),
      readRaw: vi.fn(async () => new ArrayBuffer(4)),
      save: vi.fn(async () => {}),
      saveFacts: vi.fn(async () => {}),
      addFiles: vi.fn(async () => ({ photos: [], complete: true })),
      ...library,
    },
    exporter: {
      status: vi.fn(async () => ({ available: true as const })),
      chooseDestination: vi.fn(async () => ({ token: 'test' })),
      encode: vi.fn(async () => ({ blob: new Blob(['encoded']) })),
    },
    display: { probe: vi.fn(async () => ({ supported: false, headroom: 1, accurate: true })) },
    build: { channel: 'dev', sha: 'test', date: '' },
    settingsDbName: 'xveon-test',
  };
}
