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
    render(<ExportDialog open onOpenChange={() => {}} onExport={() => {}} isExporting={false} />);

    expect(screen.getByLabelText('Ultra HDR JPEG')).toBeChecked();
    expect(screen.getByLabelText('AVIF (BT.2020 / HLG)')).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText('TIFF (Linear sRGB)'));
    expect(setFormat).toHaveBeenCalledWith('tiff');
  });

  it('calls onExport from the primary button', () => {
    const onExport = vi.fn();
    render(<ExportDialog open onOpenChange={() => {}} onExport={onExport} isExporting={false} />);
    fireEvent.click(screen.getByRole('button', { name: 'Export' }));
    expect(onExport).toHaveBeenCalledOnce();
  });

  it('disables both actions while exporting', () => {
    render(<ExportDialog open onOpenChange={() => {}} onExport={() => {}} isExporting />);
    expect(screen.getByRole('button', { name: /Exporting/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
  });

  it('really disables the quality slider for TIFF', () => {
    useAppStore.setState({ exportFormat: 'tiff' });
    const setQuality = vi.spyOn(useAppStore.getState(), 'setExportQuality');
    render(<ExportDialog open onOpenChange={() => {}} onExport={() => {}} isExporting={false} />);

    expect(document.querySelector('.xv-slider')).toHaveClass('is-disabled');
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Quality' }), { key: 'ArrowRight' });
    expect(setQuality).not.toHaveBeenCalled();
  });
});
