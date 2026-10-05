import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { ExportDialog } from './ExportDialog';
import { useAppStore } from '@/app/store';

describe('ExportDialog', () => {
  beforeEach(() => useAppStore.setState({
    files: [], selectedFileId: null, exportFormat: 'jpeg-hdr', exportQuality: 95,
  }));

  it('lists the three formats and selects one', () => {
    const setFormat = vi.spyOn(useAppStore.getState(), 'setExportFormat');
    render(<ExportDialog open onOpenChange={() => {}} onExport={() => {}} />);

    expect(screen.getByLabelText('Ultra HDR JPEG')).toBeChecked();
    expect(screen.getByLabelText('AVIF (BT.2020 / HLG)')).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText('TIFF (Linear sRGB)'));
    expect(setFormat).toHaveBeenCalledWith('tiff');
  });

  it('calls onExport from the primary button', () => {
    const onExport = vi.fn();
    render(<ExportDialog open onOpenChange={() => {}} onExport={onExport} />);
    fireEvent.click(screen.getByRole('button', { name: 'Export' }));
    expect(onExport).toHaveBeenCalledOnce();
  });

  it('closes immediately after starting export', () => {
    const onExport = vi.fn();
    const onOpenChange = vi.fn();
    render(<ExportDialog open onOpenChange={onOpenChange} onExport={onExport} />);
    fireEvent.click(screen.getByRole('button', { name: 'Export' }));
    expect(onExport).toHaveBeenCalledOnce();
    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(onExport.mock.invocationCallOrder[0]).toBeLessThan(onOpenChange.mock.invocationCallOrder[0]);
  });

  it('really disables the quality slider for TIFF', () => {
    useAppStore.setState({ exportFormat: 'tiff' });
    const setQuality = vi.spyOn(useAppStore.getState(), 'setExportQuality');
    render(<ExportDialog open onOpenChange={() => {}} onExport={() => {}} />);

    expect(document.querySelector('.xv-slider')).toHaveClass('is-disabled');
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Quality' }), { key: 'ArrowRight' });
    expect(setQuality).not.toHaveBeenCalled();
  });
});
