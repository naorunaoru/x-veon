import { useAppStore } from '@/app/store';
import { encoderFor } from '@/pipeline/export';
import { exportFormatInfo } from '@/lib/catalog';
import {
  configFromPreset, configWithOverrides, deriveHdrConfig, computeTonescaleParams,
} from '@/renderer/grading/opendrt-params';
import type { ExportFormat } from '@/lib/types';

const HDR_PEAK_LUMINANCE = 1000;

/** Grade the file's image through the app renderer and encode it. Nothing is downloaded here. */
export async function renderExport(
  fileId: string,
  format: ExportFormat = useAppStore.getState().exportFormat,
  quality: number = useAppStore.getState().exportQuality,
): Promise<{ blob: Blob; ext: string }> {
  const state = useAppStore.getState();
  const file = state.files.find((f) => f.id === fileId);
  const renderer = state.renderer;
  if (!file?.result || !renderer) throw new Error('nothing to export');

  const { exportData } = file.result;

  // Compute merged OpenDRT config (per-file preset + overrides)
  const baseConfig = configFromPreset(file.lookPreset);
  const sdrConfig = configWithOverrides(baseConfig, file.openDrtOverrides, file.preProcessOverrides);
  const sdrTs = computeTonescaleParams(sdrConfig);

  let data: Float32Array;
  let hdrData: Float32Array | null = null;
  let peakLuminance = sdrConfig.peak_luminance;

  if (exportFormatInfo(format).needsHdr) {
    // JPEG-HDR and AVIF need HDR tonemapped data
    const hdrConfig = deriveHdrConfig(sdrConfig, HDR_PEAK_LUMINANCE);
    const hdrTs = computeTonescaleParams(hdrConfig);
    peakLuminance = HDR_PEAK_LUMINANCE;

    if (format === 'jpeg-hdr') {
      // Dual render: SDR (Rec.709) + HDR (Rec.2020)
      data = await renderer.readback(sdrConfig, sdrTs, 'rec709');
      hdrData = await renderer.readback(hdrConfig, hdrTs, 'rec2020');
    } else {
      // AVIF: HDR only (Rec.2020)
      data = await renderer.readback(hdrConfig, hdrTs, 'rec2020');
    }
  } else {
    // JPEG / TIFF: SDR (Rec.709)
    data = await renderer.readback(sdrConfig, sdrTs, 'rec709');
  }

  const blob = await encoderFor(format).encode(
    data, hdrData, exportData.width, exportData.height, exportData.orientation, quality, peakLuminance,
  );
  return { blob, ext: exportFormatInfo(format).ext };
}
