import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import { useAppStore } from '@/store';
import type { ExportFormat } from '@/pipeline/types';

const FORMAT_OPTIONS: { value: ExportFormat; label: string }[] = [
  { value: 'jpeg-hdr', label: 'Ultra HDR JPEG' },
  { value: 'avif', label: 'AVIF (BT.2020 / HLG)' },
  { value: 'tiff', label: 'TIFF (Linear sRGB)' },
];

interface ExportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onExport: () => void;
  isExporting: boolean;
}

export function ExportDialog({ open, onOpenChange, onExport, isExporting }: ExportDialogProps) {
  const exportFormat = useAppStore((s) => s.exportFormat);
  const exportQuality = useAppStore((s) => s.exportQuality);
  const setExportFormat = useAppStore((s) => s.setExportFormat);
  const setExportQuality = useAppStore((s) => s.setExportQuality);

  const isTiff = exportFormat === 'tiff';

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!isExporting) onOpenChange(v); }}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Export Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-3 py-2">
          {/* Format */}
          <fieldset className="space-y-1.5">
            <span className="text-xs text-muted-foreground">Format</span>
            {FORMAT_OPTIONS.map((o) => (
              <label
                key={o.value}
                className={`flex items-center gap-2.5 rounded-md border px-3 py-2 text-xs cursor-pointer transition-colors ${
                  exportFormat === o.value
                    ? 'border-primary bg-primary/10 text-foreground'
                    : 'border-border text-muted-foreground hover:text-foreground hover:border-muted-foreground'
                }`}
              >
                <input
                  type="radio"
                  name="export-format"
                  value={o.value}
                  checked={exportFormat === o.value}
                  onChange={() => setExportFormat(o.value)}
                  className="sr-only"
                />
                {o.label}
              </label>
            ))}
          </fieldset>

          {/* Quality */}
          <div className={`flex items-center gap-3 ${isTiff ? 'opacity-40' : ''}`}>
            <span className="text-xs text-muted-foreground w-14 flex-shrink-0">Quality</span>
            <Slider
              value={[exportQuality]}
              onValueChange={([v]) => setExportQuality(v)}
              min={1}
              max={100}
              step={1}
              disabled={isTiff}
              className="flex-1"
            />
            <span className="text-xs text-muted-foreground w-7 text-right tabular-nums">
              {exportQuality}
            </span>
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)} disabled={isExporting}>
            Cancel
          </Button>
          <Button size="sm" onClick={onExport} disabled={isExporting}>
            {isExporting ? (
              <>
                <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                Exporting
              </>
            ) : (
              'Export'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
