import type { DemosaicMethod, ExportFormat, ModelSize } from '@/lib/types';
import type { Slice } from './types';

export interface SettingsSlice {
  modelSize: ModelSize;
  demosaicMethod: DemosaicMethod;
  exportFormat: ExportFormat;
  exportQuality: number;
  setModelSize: (size: ModelSize) => void;
  setDemosaicMethod: (method: DemosaicMethod) => void;
  setExportFormat: (format: ExportFormat) => void;
  setExportQuality: (quality: number) => void;
}

export const createSettingsSlice: Slice<SettingsSlice> = (set) => ({
  modelSize: 'S',
  demosaicMethod: 'neural-net',
  exportFormat: 'jpeg-hdr',
  exportQuality: 95,
  setModelSize: (modelSize) => set({ modelSize }),
  setDemosaicMethod: (demosaicMethod) => set({ demosaicMethod }),
  setExportFormat: (exportFormat) => set({ exportFormat }),
  setExportQuality: (exportQuality) => set({ exportQuality }),
});
