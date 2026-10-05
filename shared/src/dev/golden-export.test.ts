import { beforeEach, expect, it, vi } from 'vitest';
vi.mock('@/app/services/export', () => ({ renderExport: vi.fn() }));
vi.mock('@/pipeline/inference', () => ({ models: {} }));
vi.mock('@/app/services/library', () => ({ importFiles: vi.fn(), removeFile: vi.fn() }));
import { renderExport } from '@/app/services/export';
import { exportOnce } from './golden';
beforeEach(() => vi.clearAllMocks());
it('uses native receipt hashes and sizes without reading a blob', async () => {
  const read = vi.fn(); const blob = { arrayBuffer: read } as unknown as Blob;
  vi.mocked(renderExport).mockResolvedValue({ ext: 'avif', bytes: 123, sha256: 'a'.repeat(64), blob });
  await expect(exportOnce('id', 'avif')).resolves.toEqual({ bytes: 123, sha256: 'a'.repeat(64) });
  expect(read).not.toHaveBeenCalled();
});
it('hashes the blob-only web result', async () => {
  const blob = { arrayBuffer: async () => new TextEncoder().encode('abc').buffer } as Blob;
  vi.mocked(renderExport).mockResolvedValue({ ext: 'tif', blob });
  await expect(exportOnce('id', 'tiff')).resolves.toEqual({ bytes: 3, sha256: 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad' });
});
it.each([{}, { sha256: 'bad', bytes: 123 }, { sha256: 'a'.repeat(64) }, { sha256: 'a'.repeat(64), bytes: 0 }])('rejects missing or invalid receipt %j', async receipt => {
  vi.mocked(renderExport).mockResolvedValue({ ext: 'avif', ...receipt });
  await expect(exportOnce('id', 'avif')).rejects.toThrow();
});
