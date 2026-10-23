import { beforeEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto } from '@/test/fake-host';
vi.mock('@/components/OutputCanvas', () => ({ OutputCanvas: () => null }));
vi.mock('@/app/services/processing', () => ({ processFile: vi.fn() }));
import { processFile } from '@/app/services/processing';
import { PhotoStage } from './PhotoStage';
beforeEach(() => {
  vi.clearAllMocks();
  useAppStore.setState({ files: [], currentId: null, initError: null, processingFileId: null });
});
it('retries a failed photo from the error card', () => {
  const file = { ...fromLibraryPhoto(fakePhoto()), status: 'error' as const, error: 'bad RAW' };
  useAppStore.setState({ files: [file], currentId: 'a', initError: null });
  render(<PhotoStage />);
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
  expect(processFile).toHaveBeenCalledWith('a');
});
it('holds Retry while another photo is processing', () => {
  const file = { ...fromLibraryPhoto(fakePhoto()), status: 'error' as const, error: 'bad RAW' };
  useAppStore.setState({ files: [file], currentId: 'a', initError: null, processingFileId: 'b' });
  render(<PhotoStage />);
  const retry = screen.getByRole('button', { name: 'Retry' });
  expect(retry).toBeDisabled();
  fireEvent.click(retry);
  expect(processFile).not.toHaveBeenCalled();
});
