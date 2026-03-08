import { useEffect, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';
import { useProcessFile } from '@/hooks/useProcessFile';
import { useExport } from '@/hooks/useExport';
import { ExportDialog } from '@/components/ExportDialog';
import { Switch } from '@/components/ui/switch';
import type { CfaType, DemosaicMethod, ModelSize } from '@/pipeline/types';
import { getModelMeta, getAvailableSizes, switchModelSize } from '@/pipeline/inference';

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

const MODEL_SIZES: { value: ModelSize; label: string }[] = [
  { value: 'S', label: 'S' },
  { value: 'M', label: 'M' },
  { value: 'L', label: 'L' },
];

export function SettingsPanel() {
  const demosaicMethod = useAppStore((s) => s.demosaicMethod);
  const setDemosaicMethod = useAppStore((s) => s.setDemosaicMethod);
  const modelSize = useAppStore((s) => s.modelSize);
  const setModelSize = useAppStore((s) => s.setModelSize);
  const selectedFile = useAppStore((s) =>
    s.files.find((f) => f.id === s.selectedFileId),
  );
  const initialized = useAppStore((s) => s.initialized);

  const cfaType = selectedFile?.cfaType ?? null;
  const availableSizes = cfaType ? getAvailableSizes(cfaType) : new Set<ModelSize>(['S']);
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
    } else if (nextQueuedId && demosaicMethod === 'neural-net') {
      processFile(nextQueuedId);
    }
  }, [selectedFile?.id, selectedFile?.status, nextQueuedId, initialized, isProcessing, processFile, demosaicMethod]);

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

  // Auto-reprocess when ML highlight reconstruction toggle changes (NN only)
  const mlHighlightReconstruction = useAppStore((s) => s.mlHighlightReconstruction);
  const setMlHighlightReconstruction = useAppStore((s) => s.setMlHighlightReconstruction);
  const prevMlHlRef = useRef(mlHighlightReconstruction);
  useEffect(() => {
    if (prevMlHlRef.current === mlHighlightReconstruction) return;
    prevMlHlRef.current = mlHighlightReconstruction;
    if (initialized && selectedFile && (selectedFile.status === 'done' || selectedFile.status === 'error') && !isProcessing && demosaicMethod === 'neural-net') {
      processFile(selectedFile.id);
    }
  }, [mlHighlightReconstruction, initialized, selectedFile, isProcessing, processFile, demosaicMethod]);
  const { exportFile, isExporting } = useExport();

  const [exportOpen, setExportOpen] = useState(false);

  const canProcess = initialized && !isProcessing && !!selectedFile;
  const canExport = selectedFile?.status === 'done' && !isExporting;

  return (
    <div className="border-t border-border p-4 space-y-3">
      {/* Demosaic method */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-muted-foreground w-14 flex-shrink-0">Method</span>
        <div className="flex-1">
        <Select value={demosaicMethod} onValueChange={(v) => setDemosaicMethod(v as DemosaicMethod)}>
          <SelectTrigger className="h-8 text-xs">
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

      {/* Model size */}
      <section className="space-y-1.5">
        <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground">Model</h3>
        <div className="flex rounded-md bg-muted p-0.5">
          {MODEL_SIZES.map((s) => {
            const available = availableSizes.has(s.value);
            const active = modelSize === s.value;
            return (
              <button
                key={s.value}
                disabled={!available}
                className={cn(
                  'flex-1 rounded-sm px-2 py-1 text-xs font-medium transition-colors',
                  active
                    ? 'bg-background text-foreground shadow-sm'
                    : available
                      ? 'text-muted-foreground hover:text-foreground'
                      : 'text-muted-foreground/30 cursor-not-allowed',
                )}
                onClick={async () => {
                  if (active || !available) return;
                  setModelSize(s.value);
                  await switchModelSize(s.value);
                  // Reprocess current file with new model
                  const file = useAppStore.getState().files.find(
                    (f) => f.id === useAppStore.getState().selectedFileId,
                  );
                  if (file && (file.status === 'done' || file.status === 'error') && !isProcessing && demosaicMethod === 'neural-net') {
                    processFile(file.id);
                  }
                }}
              >
                {s.label}
              </button>
            );
          })}
        </div>
      </section>

      {/* ML highlight reconstruction toggle */}
      {(() => {
        const modelHasHlHead = cfaType ? getModelMeta(cfaType).hl_head === true : false;
        const canToggle = demosaicMethod === 'neural-net' && modelHasHlHead;
        return (
          <div
            className="flex items-center justify-between"
            title={!canToggle
              ? demosaicMethod !== 'neural-net'
                ? 'Only available with neural network demosaic'
                : 'Current model does not have a highlight reconstruction head'
              : 'Use ML model for highlight reconstruction instead of numeric'}
          >
            <span className={`text-xs ${canToggle ? 'text-muted-foreground' : 'text-muted-foreground/40'}`}>
              ML highlight reconstruction
            </span>
            <Switch
              checked={canToggle && mlHighlightReconstruction}
              disabled={!canToggle}
              onCheckedChange={setMlHighlightReconstruction}
            />
          </div>
        );
      })()}

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
