import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { useAppStore } from '@/store';
import { useProcessFile } from '@/hooks/useProcessFile';
import { useExport } from '@/hooks/useExport';
import type { ExportFormat } from '@/pipeline/types';

export function SettingsPanel() {
  const exportFormat = useAppStore((s) => s.exportFormat);
  const exportQuality = useAppStore((s) => s.exportQuality);
  const setExportFormat = useAppStore((s) => s.setExportFormat);
  const setExportQuality = useAppStore((s) => s.setExportQuality);
  const selectedFile = useAppStore((s) =>
    s.files.find((f) => f.id === s.selectedFileId),
  );
  const initialized = useAppStore((s) => s.initialized);

  const { processFile, isProcessing } = useProcessFile();
  const { exportFile, isExporting } = useExport();

  const canProcess = initialized && selectedFile?.status === 'queued' && !isProcessing;
  const canExport = selectedFile?.status === 'done' && !isExporting;
  const isTiff = exportFormat === 'tiff';

  return (
    <div className="border-t border-border p-4 space-y-3">
      {/* Format */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-muted-foreground w-14 flex-shrink-0">Format</span>
        <Select value={exportFormat} onValueChange={(v) => setExportFormat(v as ExportFormat)}>
          <SelectTrigger className="flex-1 h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="avif">HDR AVIF (BT.2020 / HLG)</SelectItem>
            <SelectItem value="jpeg">JPEG (sRGB)</SelectItem>
            <SelectItem value="tiff">16-bit TIFF (Linear sRGB)</SelectItem>
          </SelectContent>
        </Select>
      </div>

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

      {/* Actions */}
      <div className="flex gap-2">
        <Button
          size="sm"
          className="flex-1"
          disabled={!canProcess}
          onClick={() => selectedFile && processFile(selectedFile.id)}
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
              Processing
            </>
          ) : (
            'Process'
          )}
        </Button>
        <Button
          size="sm"
          variant="secondary"
          className="flex-1"
          disabled={!canExport}
          onClick={() => selectedFile && exportFile(selectedFile.id)}
        >
          {isExporting ? (
            <>
              <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
              Exporting
            </>
          ) : (
            'Export'
          )}
        </Button>
      </div>
    </div>
  );
}
