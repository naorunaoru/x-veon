import { useCallback, useState } from 'react';
import { useAppStore } from '@/store';
import { encodeImage } from '@/pipeline/encoder';
import { configFromPreset, configWithOverrides, deriveHdrConfig, computeTonescaleParams } from '@/gl/opendrt-params';
import type { ExportFormat } from '@/pipeline/types';

const HDR_PEAK_LUMINANCE = 1000;

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

    const renderer = state.rendererRef;
    if (!renderer) {
      console.error('Export failed: renderer not available');
      return;
    }

    const { exportFormat, exportQuality } = state;
    const { exportData } = file.result;

    // Compute merged OpenDRT config (per-file preset + overrides)
    const baseConfig = configFromPreset(file.lookPreset);
    const sdrConfig = configWithOverrides(baseConfig, file.openDrtOverrides);
    const sdrTs = computeTonescaleParams(sdrConfig);

    setIsExporting(true);

    try {
      const startTime = Date.now();

      let data: Float32Array;
      let hdrData: Float32Array | null = null;
      let peakLuminance = sdrConfig.peak_luminance;

      if (needsHdr(exportFormat)) {
        // JPEG-HDR and AVIF need HDR tonemapped data
        const hdrConfig = deriveHdrConfig(sdrConfig, HDR_PEAK_LUMINANCE);
        const hdrTs = computeTonescaleParams(hdrConfig);
        peakLuminance = HDR_PEAK_LUMINANCE;

        if (exportFormat === 'jpeg-hdr') {
          // Dual render: SDR (Rec.709) + HDR (Rec.2020)
          data = renderer.renderForExport(sdrConfig, sdrTs, 'rec709');
          hdrData = renderer.renderForExport(hdrConfig, hdrTs, 'rec2020');
        } else {
          // AVIF: HDR only (Rec.2020)
          data = renderer.renderForExport(hdrConfig, hdrTs, 'rec2020');
        }
      } else {
        // JPEG / TIFF: SDR (Rec.709)
        data = renderer.renderForExport(sdrConfig, sdrTs, 'rec709');
      }

      const { blob, ext } = await encodeImage(
        data,
        hdrData,
        exportData.width,
        exportData.height,
        exportData.orientation,
        exportFormat,
        exportQuality,
        peakLuminance,
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

function needsHdr(format: ExportFormat): boolean {
  return format === 'jpeg-hdr' || format === 'avif';
}
