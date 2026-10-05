import { beforeEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { fakeHost } from '@/test/fake-host';
import { setHost } from '@/app/services/host';
import { useExportJobs, type ExportJobView } from '@/app/services/export-jobs';
import { ExportStatus } from './ExportStatus';
let host: ReturnType<typeof fakeHost>;
const job: ExportJobView = { id: 'job', fileId: 'a', label: 'a.avif', state: 'encoding', error: null, result: null, destination: { token: 'test' } };
beforeEach(() => {
  host = fakeHost();
  host.exporter.reveal = { label: 'Show in Finder', open: vi.fn(async () => {}) };
  setHost(host);
  useExportJobs.setState({ jobs: [{ ...job }] });
});
it('announces running jobs politely and cancels them', () => {
  render(<ExportStatus />);
  expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite');
  expect(screen.getByText('Encoding a.avif…')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Cancel export of a.avif' }));
  expect(useExportJobs.getState().jobs).toEqual([]);
});
it('shows the result name, opens the destination and dismisses completed jobs', () => {
  useExportJobs.setState({ jobs: [{ ...job, state: 'done', label: 'chosen.avif' }] });
  render(<ExportStatus />);
  expect(screen.getByText('Exported chosen.avif')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Show in Finder' }));
  expect(host.exporter.reveal!.open).toHaveBeenCalledWith(job.destination);
  fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
  expect(useExportJobs.getState().jobs).toEqual([]);
});
it('shows persistent errors with Dismiss', () => {
  useExportJobs.setState({ jobs: [{ ...job, state: 'failed', error: 'disk full' }] });
  render(<ExportStatus />);
  expect(screen.getByText("Couldn't export a.avif: disk full")).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
  expect(useExportJobs.getState().jobs).toEqual([]);
});
it('omits reveal when the host has no capability or destination', () => {
  delete host.exporter.reveal;
  useExportJobs.setState({ jobs: [{ ...job, state: 'done' }] });
  render(<ExportStatus />);
  expect(screen.queryByRole('button', { name: 'Show in Finder' })).not.toBeInTheDocument();
});
it('displays reveal errors without an unhandled rejection', async () => {
  vi.mocked(host.exporter.reveal!.open).mockRejectedValue(new Error('could not open folder'));
  useExportJobs.setState({ jobs: [{ ...job, state: 'done' }] });
  render(<ExportStatus />);
  fireEvent.click(screen.getByRole('button', { name: 'Show in Finder' }));
  await waitFor(() => expect(screen.getByText('could not open folder')).toBeInTheDocument());
});
