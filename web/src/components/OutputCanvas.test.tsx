import { beforeEach, expect, it, vi } from 'vitest';
import { render, waitFor, act } from '@testing-library/react';
import type { ProcessingResultMeta } from '@/lib/types';
import type { QueuedFile } from '@/app/store';
const m = vi.hoisted(() => ({ takeResult: vi.fn(), createRenderer: vi.fn() }));
vi.mock('@/app/services/processing', () => ({ takeResult: m.takeResult }));
vi.mock('@/renderer', () => ({ createRenderer: m.createRenderer, isWebGpuSupported: () => true }));
vi.mock('@/app/hooks/usePanZoom', () => ({ usePanZoom: () => ({ transform: '', isDragging: false, handlers: {}, scale: 1 }) }));
import { OutputCanvas } from './OutputCanvas';
import { useAppStore } from '@/app/store';
const meta = { metadata: { width: 2, height: 2 }, exportData: { width: 2, height: 2, orientation: 'Normal' } } as ProcessingResultMeta;
const renderer = { setImage: vi.fn(), setGrade: vi.fn(), render: vi.fn(), dispose: vi.fn(), display: { hdr: false, headroom: 1 } };
beforeEach(() => {
  vi.resetAllMocks();
  m.createRenderer.mockResolvedValue(renderer);
  useAppStore.setState({ files: [{ id: 'a', status: 'done', result: meta, lookPreset: 'default', openDrtOverrides: {}, preProcessOverrides: {} } as QueuedFile], renderer: null, displayHdr: false });
});
it('releases the taken image and reports failed upload without publishing a renderer', async () => {
  const image = { gpu: {}, dispose: vi.fn() };
  m.takeResult.mockReturnValue(image);
  renderer.setImage.mockImplementation(() => { throw new Error('upload failed'); });
  const log = vi.spyOn(console, 'error').mockImplementation(() => {});
  render(<OutputCanvas fileId="a" result={meta} />);
  await waitFor(() => expect(useAppStore.getState().files[0].status).toBe('error'));
  expect(image.dispose).toHaveBeenCalledTimes(1);
  expect(useAppStore.getState().renderer).toBeNull();
  log.mockRestore();
});
it('does not take image ownership after unmount during renderer creation', async () => {
  let finish!: (value: typeof renderer) => void;
  m.createRenderer.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  const view = render(<OutputCanvas fileId="a" result={meta} />);
  view.unmount();
  await act(async () => { finish(renderer); });
  expect(m.takeResult).not.toHaveBeenCalled();
  expect(renderer.dispose).toHaveBeenCalledTimes(1);
});
it('uploads and releases a taken image before publishing the renderer', async () => {
  const image = { gpu: {}, dispose: vi.fn() }; m.takeResult.mockReturnValue(image);
  render(<OutputCanvas fileId="a" result={meta} />);
  await waitFor(() => expect(useAppStore.getState().renderer).toBe(renderer));
  expect(renderer.setImage).toHaveBeenCalledWith(image.gpu);
  expect(image.dispose).toHaveBeenCalledTimes(1);
});
