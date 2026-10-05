import { Slider } from '@/components/hud/Slider';
import { EXPORT_FORMATS } from '@/lib/catalog';
import { useAppStore } from '@/app/store';
import { Dialog, DialogContent } from './Dialog';

const DEFAULT_QUALITY = 95;

interface ExportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onExport: () => void;
}

export function ExportDialog({ open, onOpenChange, onExport }: ExportDialogProps) {
  const fileName = useAppStore(
    (state) => state.files.find((file) => file.id === state.selectedFileId)?.name ?? 'file',
  );
  const exportFormat = useAppStore((state) => state.exportFormat);
  const exportQuality = useAppStore((state) => state.exportQuality);
  const setExportFormat = useAppStore((state) => state.setExportFormat);
  const setExportQuality = useAppStore((state) => state.setExportQuality);
  const isTiff = exportFormat === 'tiff';

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title={`Export ${fileName}`}
        actions={(
          <>
            <button
              type="button" className="xv-btn"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </button>
            <button
              type="button" className="xv-btn xv-btn--primary"
              onClick={() => { onExport(); onOpenChange(false); }}
            >
              Export
            </button>
          </>
        )}
      >
        <fieldset className="xv-dialog__field">
          <legend className="xv-dialog__label">Format</legend>
          <div className="xv-format-list">
            {EXPORT_FORMATS.map((option) => (
              <label
                key={option.id}
                className={`xv-format${exportFormat === option.id ? ' is-selected' : ''}`}
              >
                <input
                  type="radio"
                  name="export-format"
                  value={option.id}
                  checked={exportFormat === option.id}
                  onChange={() => setExportFormat(option.id)}
                />
                {option.label}
              </label>
            ))}
          </div>
        </fieldset>
        <div className="xv-dialog__field">
          <Slider
            label="Quality"
            value={exportQuality}
            defaultValue={DEFAULT_QUALITY}
            min={1}
            max={100}
            step={1}
            onChange={setExportQuality}
            disabled={isTiff}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
