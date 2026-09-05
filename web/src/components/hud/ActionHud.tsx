import { useState } from 'react';
import { useAppStore } from '@/app/store';
import { useProcessing } from '@/app/hooks/useProcessing';
import { useExport } from '@/app/hooks/useExport';
import { ExportDialog } from '@/components/dialogs/ExportDialog';
import './ActionHud.css';

export function ActionHud() {
  const initialized = useAppStore((s) => s.initialized);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const { processFile, isProcessing } = useProcessing();
  const { exportFile, isExporting } = useExport();
  const [exportOpen, setExportOpen] = useState(false);

  const canProcess = initialized && !isProcessing && !!selectedFile;
  const canExport = selectedFile?.status === 'done' && !isExporting;

  return (
    <div className="xv-actionhud xv-glass">
      <button
        className="xv-action-btn"
        disabled={!canExport}
        onClick={() => setExportOpen(true)}
      >
        Export
      </button>
      <button
        className="xv-action-btn is-primary"
        disabled={!canProcess}
        onClick={() => selectedFile && processFile(selectedFile.id)}
      >
        {isProcessing ? 'Processing…' : 'Process'}
      </button>
      <ExportDialog
        open={exportOpen}
        onOpenChange={setExportOpen}
        onExport={() => selectedFile && exportFile(selectedFile.id)}
        isExporting={isExporting}
      />
    </div>
  );
}
