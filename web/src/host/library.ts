import type {
  LibraryHost,
  LibraryPhoto,
  LibraryChange,
  LibrarySnapshot,
  PhotoEdit,
  PhotoFacts,
} from '@/host';
import { RAW_EXTENSIONS } from '@/lib/catalog';
import { extractRafThumbnail, extractRafQuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import { createFileStorage, fromRecord, toRecord, type PersistedFile } from './idb-storage';
import { createOpfsStorage } from './opfs-storage';
import { deleteDatabase, assertDatabaseActive, onDatabaseVersionChange } from '@/app/storage/database';

export function createWebLibrary(options: {
  dbName: string;
  opfsRoot: string;
  reload?: () => void;
}): LibraryHost {
  const records = createFileStorage(options.dbName);
  const opfs = createOpfsStorage(options.opfsRoot);
  const photos = new Map<string, LibraryPhoto>();
  const persisted = new Map<string, PersistedFile>();
  const files = new Map<string, File>();
  const imports = new Map<string, Promise<void>>();
  const operations = new Set<Promise<unknown>>();
  const removed = new Set<string>();
  const listeners = new Set<(change: LibraryChange) => void>();
  let clearing = false;
  let invalidated = false;
  onDatabaseVersionChange(options.dbName, {
    suspend: () => {
      invalidated = true;
      return Promise.allSettled([...operations]).then(() => {});
    },
    closed: () => options.reload?.(),
  });
  const track = <T>(promise: Promise<T>): Promise<T> => {
    operations.add(promise);
    void promise.then(
      () => operations.delete(promise),
      () => operations.delete(promise),
    );
    return promise;
  };
  function release(items: { thumbnailUrl: string | null }[]) {
    for (const item of items)
      if (item.thumbnailUrl?.startsWith('blob:')) URL.revokeObjectURL(item.thumbnailUrl);
  }
  function publish(photo: LibraryPhoto) {
    for (const listener of listeners)
      listener({ snapshot: { photos: [photo], complete: true }, kind: 'facts' });
  }
  async function load(): Promise<LibrarySnapshot> {
    assertDatabaseActive(options.dbName);
    if (clearing) throw new Error('Library is being cleared.');
    let rows: PersistedFile[];
    try {
      rows = await records.getAllFiles();
    } catch (error) {
      console.warn('Could not read the library from IndexedDB:', error);
      return { photos: [...photos.values()], complete: false };
    }
    for (const row of rows) {
      if (removed.has(row.id) || files.has(row.id)) continue;
      persisted.set(row.id, row);
      const thumb = await opfs.readThumbnail(row.id).catch(() => null);
      if (removed.has(row.id) || clearing || invalidated) continue;
      const existing = photos.get(row.id);
      if (existing) release([existing]);
      photos.set(row.id, fromRecord(row, thumb ? URL.createObjectURL(thumb) : null));
    }
    assertDatabaseActive(options.dbName);
    // Include imports made while IDB or OPFS reads were pending.
    try {
      for (const id of await opfs.listRawFileIds()) {
        if (!photos.has(id) && !files.has(id)) await opfs.deleteAllForFile(id);
      }
    } catch {
      /* Non-critical cleanup, as before. */
    }
    return { photos: [...photos.values()], complete: true };
  }
  async function importOne(photo: LibraryPhoto, file: File): Promise<void> {
    const live = () => !removed.has(photo.id) && !clearing && !invalidated;
    try {
      const bytes = await file.arrayBuffer();
      if (!live()) return;
      const thumbnail = extractRafThumbnail(bytes);
      const metadata = extractRafQuickMetadata(bytes);
      const current = photos.get(photo.id)!;
      const updated = {
        ...current,
        thumbnailUrl: thumbnail ? URL.createObjectURL(thumbnail) : null,
        facts: { ...current.facts, metadata },
      };
      photos.set(photo.id, updated);
      publish(updated);
      await Promise.all([
        opfs.writeRaw(photo.id, bytes).catch((error) => console.warn('OPFS raw write failed:', error)),
        thumbnail
          ? opfs
              .writeThumbnail(photo.id, thumbnail)
              .catch((error) => console.warn('OPFS thumbnail write failed:', error))
          : undefined,
      ]);
      if (!live()) return;
      const record = toRecord(updated, updated.edit, updated.facts, Date.now());
      persisted.set(photo.id, record);
      try {
        await records.putFile(record);
      } catch (error) {
        if (!live()) return;
        const failed = {
          ...updated,
          editing: 'session' as const,
          editingNote: error instanceof Error ? error.message : String(error),
        };
        photos.set(photo.id, failed);
        publish(failed);
      }
    } catch (error) {
      console.warn('RAW import failed:', error);
    }
  }
  async function addFiles(dropped: File[]): Promise<LibrarySnapshot> {
    assertDatabaseActive(options.dbName);
    if (clearing) throw new Error('Library is being cleared.');
    const added: LibraryPhoto[] = [];
    for (const file of dropped) {
      if (!RAW_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext))) continue;
      const photo: LibraryPhoto = {
        id: crypto.randomUUID(),
        name: file.name.replace(/\.[^.]+$/, ''),
        originalName: file.name,
        fileSize: file.size,
        thumbnailUrl: null,
        editing: 'saved',
        editingNote: null,
        edit: {
          version: 1,
          lookPreset: 'default',
          openDrtOverrides: {},
          preProcessOverrides: {},
          demosaicMethod: null,
          model: null,
        },
        facts: {
          cfaType: file.name.toLowerCase().endsWith('.raf') ? 'xtrans' : 'bayer',
          metadata: null,
          resultMeta: null,
          lensProfile: null,
          status: 'queued',
          error: null,
        },
      };
      photos.set(photo.id, photo);
      files.set(photo.id, file);
      added.push(photo);
      const work = track(importOne(photo, file));
      imports.set(photo.id, work);
      void work.finally(() => imports.delete(photo.id));
    }
    await Promise.resolve();
    return {
      photos: added.map((p) => photos.get(p.id) ?? p),
      complete: true,
      selectedIds: added.length ? [added[0].id] : [],
    };
  }
  async function save(id: string, edit: PhotoEdit, facts: PhotoFacts): Promise<void> {
    assertDatabaseActive(options.dbName);
    if (clearing || removed.has(id)) throw new Error('Photo is no longer in the library.');
    await imports.get(id);
    assertDatabaseActive(options.dbName);
    if (clearing || removed.has(id)) throw new Error('Photo is no longer in the library.');
    const photo = photos.get(id);
    if (!photo) throw new Error('Photo is no longer in the library.');
    const old = persisted.get(id);
    const record = {
      ...old,
      ...toRecord(photo, edit, facts, old?.addedAt ?? Date.now()),
      cachedMethods: old?.cachedMethods ?? [],
    };
    await records.putFile(record);
    persisted.set(id, record);
    photos.set(id, { ...photo, edit, facts });
  }
  async function remove(id: string): Promise<void> {
    assertDatabaseActive(options.dbName);
    removed.add(id);
    const photo = photos.get(id);
    if (photo) release([photo]);
    photos.delete(id);
    files.delete(id);
    persisted.delete(id);
    await imports.get(id);
    // Shared persistence drains its per-photo save before invoking remove.
    await Promise.all([opfs.deleteAllForFile(id), records.deleteFile(id)]);
  }
  async function clear(): Promise<void> {
    assertDatabaseActive(options.dbName);
    clearing = true;
    try {
      await Promise.allSettled([...operations]);
      await deleteDatabase(options.dbName);
      const legacy = options.opfsRoot === 'stable';
      if (legacy) await deleteDatabase('xtrans-demosaic');
      await opfs.clear(legacy);
      release([...photos.values()]);
      photos.clear();
      files.clear();
      persisted.clear();
      removed.clear();
      options.reload?.();
    } finally {
      release([...photos.values()]);
      photos.clear();
      files.clear();
      persisted.clear();
      removed.clear();
      clearing = false;
    }
  }
  return {
    load: () => track(load()),
    addFiles,
    readRaw: async (id) => {
      assertDatabaseActive(options.dbName);
      const file = files.get(id);
      if (file) return file.arrayBuffer();
      const bytes = await opfs.readRaw(id);
      if (!bytes) throw new Error('RAW file not found in storage. Please re-add this file.');
      return bytes;
    },
    save: (id, edit, facts) => track(save(id, edit, facts)),
    remove: (id) => track(remove(id)),
    clear,
    onChange: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    release: (items) => {
      // A cancelled StrictMode bootstrap can receive the same current snapshot as the live one.
      // Current URLs stay owned by the host until replacement, removal or clear.
      release(
        items.filter(
          (item) => ![...photos.values()].some((photo) => photo.thumbnailUrl === item.thumbnailUrl),
        ),
      );
    },
  };
}
