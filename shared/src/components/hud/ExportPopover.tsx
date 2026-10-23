import type { Ref } from 'react';
import { Slider } from '@/components/hud/Slider';
import { EXPORT_FORMATS } from '@/lib/catalog';
import { useAppStore } from '@/app/store';
import { getHost } from '@/app/services/host';
import { exportTargets, startExport } from '@/app/services/export-batch';
import './ExportPopover.css';

const DEFAULT_QUALITY = 95;

interface ExportPopoverProps {
  popoverProps: { ref: Ref<HTMLDivElement>; role: string; tabIndex: number };
  onClose: () => void;
  exportDisabled: boolean;
}

/** The export settings, anchored above the column's foot; every change goes straight to the store. */
export function ExportPopover({ popoverProps, onClose, exportDisabled }: ExportPopoverProps) {
  const selectedCount = useAppStore((s) => s.selectedIds.length);
  const exportFormat = useAppStore((s) => s.exportFormat);
  const exportQuality = useAppStore((s) => s.exportQuality);
  const saveTo = useAppStore((s) => s.exportSaveTo);
  const onExists = useAppStore((s) => s.exportOnExists);
  const setExportFormat = useAppStore((s) => s.setExportFormat);
  const setExportQuality = useAppStore((s) => s.setExportQuality);
  const setExportSaveTo = useAppStore((s) => s.setExportSaveTo);
  const setExportOnExists = useAppStore((s) => s.setExportOnExists);
  const chooseFolder = getHost().exporter.chooseFolder;
  // A cancelled pick (null) leaves the stored choice as it was.
  const pickFolder = () => {
    chooseFolder?.()
      .then((path) => { if (path) setExportSaveTo({ mode: 'folder', path }); })
      .catch((error: unknown) => console.error('Choosing the export folder failed', error));
  };
  const onExport = () => {
    startExport(exportTargets(useAppStore.getState()));
    onClose();
  };
  return (
    <div {...popoverProps} className="xv-export-popover xv-glass-heavy" aria-label="Export settings">
      <fieldset className="xv-export-popover__field">
        <legend className="xv-export-popover__label">Format</legend>
        <div className="xv-format-list">
          {EXPORT_FORMATS.map((option) => (
            <label key={option.id} className={`xv-format${exportFormat === option.id ? ' is-selected' : ''}`}>
              <input
                type="radio" name="export-format" value={option.id}
                checked={exportFormat === option.id} onChange={() => setExportFormat(option.id)}
              />
              {option.label}
            </label>
          ))}
        </div>
      </fieldset>
      <div className="xv-export-popover__field">
        <Slider
          label="Quality" value={exportQuality} defaultValue={DEFAULT_QUALITY} min={1} max={100} step={1}
          onChange={setExportQuality} disabled={exportFormat === 'tiff'}
        />
      </div>
      {chooseFolder && (
        <>
          <fieldset className="xv-export-popover__field">
            <legend className="xv-export-popover__label">Save to</legend>
            <div className="xv-format-list">
              <label className={`xv-format${saveTo.mode === 'beside' ? ' is-selected' : ''}`}>
                <input
                  type="radio" name="export-save-to" value="beside"
                  checked={saveTo.mode === 'beside'} onChange={() => setExportSaveTo({ mode: 'beside' })}
                />
                Next to the RAWs
              </label>
              <label className={`xv-format${saveTo.mode === 'folder' ? ' is-selected' : ''}`}>
                <input type="radio" name="export-save-to" value="folder" checked={saveTo.mode === 'folder'} onChange={pickFolder} />
                Folder…
              </label>
            </div>
            {saveTo.mode === 'folder' && (
              <div className="xv-export-popover__path">
                <span title={saveTo.path}>{saveTo.path}</span>
                <button type="button" className="xv-btn" onClick={pickFolder}>Change…</button>
              </div>
            )}
          </fieldset>
          <fieldset className="xv-export-popover__field">
            <legend className="xv-export-popover__label">If the file exists</legend>
            <div className="xv-format-list xv-format-list--row">
              {(['skip', 'replace'] as const).map((value) => (
                <label key={value} className={`xv-format${onExists === value ? ' is-selected' : ''}`}>
                  <input
                    type="radio" name="export-on-exists" value={value}
                    checked={onExists === value} onChange={() => setExportOnExists(value)}
                  />
                  {value === 'skip' ? 'Skip' : 'Replace'}
                </label>
              ))}
            </div>
          </fieldset>
        </>
      )}
      <div className="xv-export-popover__actions">
        <button type="button" className="xv-btn xv-btn--primary" disabled={exportDisabled} onClick={onExport}>
          {selectedCount > 1 ? `Export ${selectedCount}` : 'Export'}
        </button>
      </div>
    </div>
  );
}
