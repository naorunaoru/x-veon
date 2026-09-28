import { getHost } from '@/app/services/host';
import { useCallback, useEffect, useState } from 'react';
import { enqueueExport } from '@/app/services/export';
export function useExport() {
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportAvailable, setExportAvailable] = useState(false);
  useEffect(() => {
    let cancelled = false;
    void getHost()
      .exporter.status()
      .then((status) => {
        if (cancelled) return;
        setExportAvailable(status.available);
        if (!status.available) setExportError(status.reason);
      })
      .catch((error) => {
        if (!cancelled) setExportError(error instanceof Error ? error.message : String(error));
      });
    return () => {
      cancelled = true;
    };
  }, []);
  const exportFile = useCallback(async (fileId: string) => {
    setIsExporting(true);
    setExportError(null);
    try {
      await enqueueExport(fileId).promise;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setExportError(message);
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  }, []);
  return { exportFile, isExporting, exportError, exportAvailable };
}
