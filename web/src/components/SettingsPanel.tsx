import { useEffect, useRef, useState } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';
import { useProcessFile } from '@/hooks/useProcessFile';
import { useExport } from '@/hooks/useExport';
import { ExportDialog } from '@/components/ExportDialog';
import type { CfaType, DemosaicMethod, ModelSize } from '@/pipeline/types';
import { getActiveModelKey, getAvailableModels, getAvailableSizes, refreshManifest, setActiveModel, switchModelSize } from '@/pipeline/inference';

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
  const selectedModelKeys = useAppStore((s) => s.selectedModelKeys);
  const setSelectedModelKey = useAppStore((s) => s.setSelectedModelKey);
  const selectedFile = useAppStore((s) =>
    s.files.find((f) => f.id === s.selectedFileId),
  );
  const initialized = useAppStore((s) => s.initialized);
  const [manifestRevision, setManifestRevision] = useState(0);
  const [isRefreshingModels, setIsRefreshingModels] = useState(false);

  const cfaType = selectedFile?.cfaType ?? null;
  const availableSizes = cfaType ? getAvailableSizes(cfaType) : new Set<ModelSize>(['S']);
  const availableMethods = DEMOSAIC_OPTIONS.filter(
    (o) => !o.cfa || !cfaType || o.cfa === cfaType,
  );
  const availableModels = cfaType ? getAvailableModels(cfaType, modelSize) : [];
  const currentModelKey = cfaType
    ? (selectedModelKeys[cfaType] ?? getActiveModelKey(cfaType) ?? '')
    : '';
  void manifestRevision;

  // Auto-fallback if current method isn't available for this CFA type
  useEffect(() => {
    if (!availableMethods.some((o) => o.value === demosaicMethod)) {
      setDemosaicMethod('neural-net');
    }
  }, [availableMethods, demosaicMethod, setDemosaicMethod]);

  useEffect(() => {
    if (!cfaType) return;
    if (availableModels.length === 0) {
      if (selectedModelKeys[cfaType]) {
        setSelectedModelKey(cfaType, null);
      }
      return;
    }
    if (currentModelKey && availableModels.some((model) => model.key === currentModelKey)) {
      return;
    }
    const fallbackKey = availableModels[0].key;
    setSelectedModelKey(cfaType, fallbackKey);
    void setActiveModel(cfaType, fallbackKey).catch((err) => {
      console.warn('Failed to activate fallback model:', err);
    });
  }, [availableModels, cfaType, currentModelKey, selectedModelKeys, setSelectedModelKey]);

  const { processFile, isProcessing } = useProcessFile();

  // Auto-process: process the selected file when it's queued (fresh drop or restored)
  useEffect(() => {
    if (!initialized || isProcessing) return;
    if (selectedFile?.status === 'queued') {
      processFile(selectedFile.id);
    }
  }, [selectedFile?.id, selectedFile?.status, initialized, isProcessing, processFile]);

  // Auto-reprocess on setting changes, but not when merely switching to a different file.
  const prevSelectedFileIdRef = useRef<string | null>(selectedFile?.id ?? null);
  const prevMethodRef = useRef(demosaicMethod);
  useEffect(() => {
    const selectedFileId = selectedFile?.id ?? null;
    if (prevSelectedFileIdRef.current !== selectedFileId) {
      prevSelectedFileIdRef.current = selectedFileId;
      prevMethodRef.current = demosaicMethod;
      return;
    }
    if (prevMethodRef.current === demosaicMethod) return;
    prevMethodRef.current = demosaicMethod;
    if (initialized && selectedFile && (selectedFile.status === 'done' || selectedFile.status === 'error') && !isProcessing) {
      processFile(selectedFile.id);
    }
  }, [demosaicMethod, initialized, selectedFile, isProcessing, processFile]);

  const prevModelKeyRef = useRef(currentModelKey);
  useEffect(() => {
    const selectedFileId = selectedFile?.id ?? null;
    if (prevSelectedFileIdRef.current !== selectedFileId) {
      prevSelectedFileIdRef.current = selectedFileId;
      prevModelKeyRef.current = currentModelKey;
      return;
    }
    if (!cfaType || demosaicMethod !== 'neural-net') {
      prevModelKeyRef.current = currentModelKey;
      return;
    }
    if (prevModelKeyRef.current === currentModelKey) return;
    prevModelKeyRef.current = currentModelKey;
    if (initialized && selectedFile && (selectedFile.status === 'done' || selectedFile.status === 'error') && !isProcessing) {
      processFile(selectedFile.id);
    }
  }, [currentModelKey, cfaType, demosaicMethod, initialized, selectedFile, isProcessing, processFile]);

  const { exportFile, isExporting } = useExport();

  const [exportOpen, setExportOpen] = useState(false);

  const canProcess = initialized && !isProcessing && !!selectedFile;
  const canExport = selectedFile?.status === 'done' && !isExporting;
  const canSelectModel = demosaicMethod === 'neural-net' && !!cfaType && availableModels.length > 0;

  async function handleModelChange(key: string) {
    if (!cfaType) return;
    setSelectedModelKey(cfaType, key);
    await setActiveModel(cfaType, key);
  }

  async function handleRefreshModels() {
    setIsRefreshingModels(true);
    try {
      await refreshManifest();
      setManifestRevision((v) => v + 1);
      if (cfaType && currentModelKey && getAvailableModels(cfaType, modelSize).some((model) => model.key === currentModelKey)) {
        await setActiveModel(cfaType, currentModelKey);
      }
    } finally {
      setIsRefreshingModels(false);
    }
  }

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

      {/* Model selector */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-muted-foreground w-14 flex-shrink-0">Model</span>
        <div className="flex-1">
          <Select
            value={currentModelKey || undefined}
            onValueChange={(value) => { void handleModelChange(value); }}
            disabled={!canSelectModel}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue placeholder={cfaType ? 'No model' : 'Select file'} />
            </SelectTrigger>
            <SelectContent>
              {availableModels.map(({ key, meta }) => (
                <SelectItem key={key} value={key}>
                  {meta.checkpoint_version ?? key}{meta.registry_status === 'beta' ? ' (beta)' : ''}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button
          size="icon"
          variant="outline"
          className="h-8 w-8 flex-shrink-0"
          disabled={isRefreshingModels}
          onClick={() => { void handleRefreshModels(); }}
          title="Refresh model manifest"
        >
          <RefreshCw className={cn('h-3.5 w-3.5', isRefreshingModels && 'animate-spin')} />
        </Button>
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
                  if (cfaType) {
                    const preferredKey = useAppStore.getState().selectedModelKeys[cfaType];
                    if (preferredKey && getAvailableModels(cfaType, s.value).some((model) => model.key === preferredKey)) {
                      await setActiveModel(cfaType, preferredKey);
                    }
                  }
                }}
              >
                {s.label}
              </button>
            );
          })}
        </div>
      </section>

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
