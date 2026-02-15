import { useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { useAppStore } from '@/store';
import { useProcessFile } from '@/hooks/useProcessFile';
import { useExport } from '@/hooks/useExport';
import type { CfaType, DemosaicMethod, ExportFormat } from '@/pipeline/types';

const DEMOSAIC_OPTIONS: { value: DemosaicMethod; label: string; cfa?: CfaType }[] = [
  { value: 'neural-net', label: 'Neural Network' },
  // X-Trans
  { value: 'markesteijn3', label: 'Markesteijn (3-pass)', cfa: 'xtrans' },
  { value: 'markesteijn1', label: 'Markesteijn (1-pass)', cfa: 'xtrans' },
  { value: 'dht', label: 'DHT (GPU)', cfa: 'xtrans' },
  // Bayer
  { value: 'ahd', label: 'AHD', cfa: 'bayer' },
  { value: 'ppg', label: 'PPG', cfa: 'bayer' },
  { value: 'mhc', label: 'MHC', cfa: 'bayer' },
  // Both
  { value: 'bilinear', label: 'Bilinear' },
];

export function SettingsPanel() {
  const demosaicMethod = useAppStore((s) => s.demosaicMethod);
  const setDemosaicMethod = useAppStore((s) => s.setDemosaicMethod);
  const exportFormat = useAppStore((s) => s.exportFormat);
  const exportQuality = useAppStore((s) => s.exportQuality);
  const setExportFormat = useAppStore((s) => s.setExportFormat);
  const setExportQuality = useAppStore((s) => s.setExportQuality);
  const selectedFile = useAppStore((s) =>
    s.files.find((f) => f.id === s.selectedFileId),
  );
  const initialized = useAppStore((s) => s.initialized);

  const cfaType = selectedFile?.cfaType ?? null;
  const availableMethods = DEMOSAIC_OPTIONS.filter(
    (o) => !o.cfa || !cfaType || o.cfa === cfaType,
  );

  // Auto-fallback if current method isn't available for this CFA type
  useEffect(() => {
    if (!availableMethods.some((o) => o.value === demosaicMethod)) {
      setDemosaicMethod('neural-net');
    }
  }, [cfaType]);

  const { processFile, isProcessing } = useProcessFile();
  const { exportFile, isExporting } = useExport();

  // [ / ] keyboard shortcuts to cycle demosaic method and auto-reprocess
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === '[' || e.key === ']') {
        e.preventDefault();
        const dir = e.key === '[' ? -1 : 1;
        // Read latest available methods + current method from store
        const method = useAppStore.getState().demosaicMethod;
        const idx = availableMethods.findIndex((o) => o.value === method);
        const next = availableMethods[(idx + dir + availableMethods.length) % availableMethods.length];
        if (next && next.value !== method) {
          setDemosaicMethod(next.value);
          // Auto-process if a file is selected and not already processing
          const state = useAppStore.getState();
          const file = state.files.find((f) => f.id === state.selectedFileId);
          if (file && state.initialized && file.status !== 'processing') {
            // Defer to next tick so the store update is committed
            setTimeout(() => processFile(file.id), 0);
          }
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [availableMethods, setDemosaicMethod, processFile]);

  const canProcess = initialized && !isProcessing;
  const canExport = selectedFile?.status === 'done' && !isExporting;
  const isTiff = exportFormat === 'tiff';

  return (
    <div className="border-t border-border p-4 space-y-3">
      {/* Demosaic method */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-muted-foreground w-14 flex-shrink-0">Method</span>
        <Select value={demosaicMethod} onValueChange={(v) => setDemosaicMethod(v as DemosaicMethod)}>
          <SelectTrigger className="flex-1 h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {availableMethods.map((o) => (
              <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

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
