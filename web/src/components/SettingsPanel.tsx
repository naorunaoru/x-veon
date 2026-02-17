import { useEffect, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import { useAppStore } from '@/store';
import { useProcessFile } from '@/hooks/useProcessFile';
import { useExport } from '@/hooks/useExport';
import { ExportDialog } from '@/components/ExportDialog';
import type { CfaType, DemosaicMethod } from '@/pipeline/types';

const DEMOSAIC_OPTIONS: { value: DemosaicMethod; label: string; cfa?: CfaType }[] = [
  { value: 'neural-net', label: 'X-veon' },
  // X-Trans
  { value: 'markesteijn3', label: 'Markesteijn (3-pass)', cfa: 'xtrans' },
  { value: 'markesteijn1', label: 'Markesteijn (1-pass)', cfa: 'xtrans' },
  { value: 'dht', label: 'DHT', cfa: 'xtrans' },
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

  // Auto-process: work through queued files sequentially
  const nextQueuedId = useAppStore((s) => s.files.find((f) => f.status === 'queued')?.id ?? null);
  useEffect(() => {
    if (!initialized || isProcessing) return;
    if (selectedFile?.status === 'queued') {
      processFile(selectedFile.id);
    } else if (nextQueuedId) {
      processFile(nextQueuedId);
    }
  }, [selectedFile?.id, selectedFile?.status, nextQueuedId, initialized, isProcessing, processFile]);

  // Auto-reprocess on method change (restore from cache if available)
  const restoreCachedResult = useAppStore((s) => s.restoreCachedResult);
  const prevMethodRef = useRef(demosaicMethod);
  useEffect(() => {
    if (prevMethodRef.current === demosaicMethod) return;
    prevMethodRef.current = demosaicMethod;
    if (initialized && selectedFile && (selectedFile.status === 'done' || selectedFile.status === 'error') && !isProcessing) {
      if (selectedFile.cachedResults[demosaicMethod]) {
        restoreCachedResult(selectedFile.id, demosaicMethod);
      } else {
        processFile(selectedFile.id);
      }
    }
  }, [demosaicMethod, initialized, selectedFile, isProcessing, processFile, restoreCachedResult]);
  const { exportFile, isExporting } = useExport();

  const [exportOpen, setExportOpen] = useState(false);

  const canProcess = initialized && !isProcessing && !!selectedFile;
  const canExport = selectedFile?.status === 'done' && !isExporting;

  return (
    <div className="border-t border-border p-4 space-y-3">
      {/* Demosaic method */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-muted-foreground w-14 flex-shrink-0">Method</span>
        <div className="flex-1 flex rounded-md bg-muted p-0.5">
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
          onClick={() => setExportOpen(true)}
        >
          Export
        </Button>
      </div>

      <ExportDialog
        open={exportOpen}
        onOpenChange={setExportOpen}
        onExport={() => selectedFile && exportFile(selectedFile.id)}
        isExporting={isExporting}
      />
    </div>
  );
}
