import { beforeEach, expect, it, vi } from 'vitest';
import { render, waitFor, act } from '@testing-library/react';
import type { ProcessingResultMeta } from '@/lib/types';
import type { QueuedFile } from '@/app/store';
const m = vi.hoisted(() => ({ acquireResult: vi.fn(), createRenderer: vi.fn() }));
vi.mock('@/app/services/processing', () => ({ acquireResult: m.acquireResult }));
vi.mock('@/renderer', () => ({ createRenderer: m.createRenderer, isWebGpuSupported: () => true }));
vi.mock('@/app/hooks/usePanZoom', () => ({ usePanZoom: () => ({ transform: '', isDragging: false, handlers: {}, scale: 1 }) }));
import { OutputCanvas } from './OutputCanvas';
import { useAppStore } from '@/app/store';
const meta = { metadata: { width: 2, height: 2 }, exportData: { width: 2, height: 2, orientation: 'Normal' } } as ProcessingResultMeta;
const renderer = {
  setImage: vi.fn(), setGrade: vi.fn(), render: vi.fn(), dispose: vi.fn(),
  setDisplay: vi.fn((d: { hdr: boolean; headroom: number }) => { renderer.display = d; }),
  display: { hdr: false, headroom: 1 },
};
const lease = () => ({ image: { gpu: {} }, release: vi.fn() });
beforeEach(() => {
  vi.resetAllMocks();
  renderer.display = { hdr: false, headroom: 1 };
  renderer.setDisplay.mockImplementation((d) => { renderer.display = d; });
  m.createRenderer.mockResolvedValue(renderer);
  useAppStore.setState({ files: [{ id: 'a', status: 'done', result: meta, lookPreset: 'default', openDrtOverrides: {}, preProcessOverrides: {} } as QueuedFile], renderer: null, displayHdr: false, displayHdrHeadroom: 1 });
});
it('gives the borrowed image back and reports a failed upload without publishing a renderer', async () => {
  const l = lease(); m.acquireResult.mockReturnValue(l);
  renderer.setImage.mockImplementation(() => { throw new Error('upload failed'); });
  const log = vi.spyOn(console, 'error').mockImplementation(() => {});
  render(<OutputCanvas fileId="a" result={meta} />);
  await waitFor(() => expect(useAppStore.getState().files[0].status).toBe('error'));
  expect(l.release).toHaveBeenCalledTimes(1);
  expect(useAppStore.getState().renderer).toBeNull();
  log.mockRestore();
});
it('does not borrow the image after unmount during renderer creation', async () => {
  let finish!: (value: typeof renderer) => void;
  m.createRenderer.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  const view = render(<OutputCanvas fileId="a" result={meta} />);
  view.unmount();
  await act(async () => { finish(renderer); });
  expect(m.acquireResult).not.toHaveBeenCalled();
  expect(renderer.dispose).toHaveBeenCalledTimes(1);
});
it('shows a borrowed image, keeps it while shown and gives it back on unmount', async () => {
  const l = lease(); m.acquireResult.mockReturnValue(l);
  const view = render(<OutputCanvas fileId="a" result={meta} />);
  await waitFor(() => expect(useAppStore.getState().renderer).toBe(renderer));
  expect(renderer.setImage).toHaveBeenCalledWith(l.image.gpu);
  expect(l.release).not.toHaveBeenCalled();
  view.unmount();
  expect(l.release).toHaveBeenCalledTimes(1);
  expect(renderer.dispose).toHaveBeenCalledTimes(1);
});
it('re-queues the file when no result is in memory', async () => {
  m.acquireResult.mockReturnValue(null);
  render(<OutputCanvas fileId="a" result={meta} />);
  await waitFor(() => expect(useAppStore.getState().files[0].status).toBe('queued'));
  expect(renderer.setImage).not.toHaveBeenCalled();
});
it('switches to HDR in place: no new renderer, no reprocessing', async () => {
  const l = lease(); m.acquireResult.mockReturnValue(l);
  render(<OutputCanvas fileId="a" result={meta} />);
  await waitFor(() => expect(useAppStore.getState().renderer).toBe(renderer));
  act(() => { useAppStore.setState({ displayHdr: true, displayHdrHeadroom: 4 }); });
  expect(renderer.setDisplay).toHaveBeenCalledWith({ hdr: true, headroom: 4 });
  expect(m.createRenderer).toHaveBeenCalledTimes(1);
  expect(m.acquireResult).toHaveBeenCalledTimes(1);
  expect(useAppStore.getState().files[0].status).toBe('done');
});
it('swaps to a new result for the same file and gives the previous one back', async () => {
  const first = lease(), second = lease();
  m.acquireResult.mockReturnValueOnce(first).mockReturnValueOnce(second);
  const view = render(<OutputCanvas fileId="a" result={meta} />);
  await waitFor(() => expect(renderer.setImage).toHaveBeenCalledWith(first.image.gpu));
  view.rerender(<OutputCanvas fileId="a" result={{ ...meta }} />);
  await waitFor(() => expect(renderer.setImage).toHaveBeenCalledWith(second.image.gpu));
  expect(first.release).toHaveBeenCalledTimes(1);
  expect(second.release).not.toHaveBeenCalled();
  expect(m.createRenderer).toHaveBeenCalledTimes(1);
});
