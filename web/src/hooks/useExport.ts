import { useCallback, useState } from 'react';
import { useAppStore } from '@/store';
import { encodeImage } from '@/pipeline/encoder';
import { readHwc, hwcKey } from '@/lib/opfs-storage';

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

export function useExport() {
  const [isExporting, setIsExporting] = useState(false);

  const exportFile = useCallback(async (fileId: string) => {
    const state = useAppStore.getState();
    const file = state.files.find((f) => f.id === fileId);
    if (!file?.result) return;

    const { exportFormat, exportQuality, lookPreset } = state;
    const { exportData } = file.result;

    setIsExporting(true);

    try {
      const hwc = await readHwc(file.resultMethod ? hwcKey(fileId, file.resultMethod) : fileId);
      if (!hwc) throw new Error('Image data not found. The file may need to be reprocessed.');

      const startTime = Date.now();
      const { blob, ext } = await encodeImage(
        hwc,
        exportData.width,
        exportData.height,
        exportData.xyzToCam,
        exportData.wbCoeffs,
        exportData.orientation,
        exportFormat,
        exportQuality,
        lookPreset,
      );
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`Exported ${exportFormat.toUpperCase()} - ${(blob.size / 1024 / 1024).toFixed(1)} MB in ${elapsed}s`);
      triggerDownload(blob, `${file.name}.${ext}`);
    } catch (e) {
      console.error('Export failed:', e);
    } finally {
      setIsExporting(false);
    }
  }, []);

  return { exportFile, isExporting };
}
