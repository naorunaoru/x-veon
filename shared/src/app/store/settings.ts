import type { DemosaicMethod, ExportFormat, ModelSize } from '@/lib/types';
import type { OnExists, SaveTo } from '@/host';
import type { Slice } from './types';

export interface SettingsSlice {
  modelSize: ModelSize;
  demosaicMethod: DemosaicMethod;
  exportFormat: ExportFormat;
  exportQuality: number;
  exportSaveTo: SaveTo;
  exportOnExists: OnExists;
  setModelSize: (size: ModelSize) => void;
  setDemosaicMethod: (method: DemosaicMethod) => void;
  setExportFormat: (format: ExportFormat) => void;
  setExportQuality: (quality: number) => void;
  setExportSaveTo: (saveTo: SaveTo) => void;
  setExportOnExists: (onExists: OnExists) => void;
}

export const createSettingsSlice: Slice<SettingsSlice> = (set) => ({
  modelSize: 'S',
  demosaicMethod: 'neural-net',
  exportFormat: 'jpeg-hdr',
  exportQuality: 95,
  exportSaveTo: { mode: 'beside' },
  exportOnExists: 'skip',
  setModelSize: (modelSize) => set({ modelSize }),
  setDemosaicMethod: (demosaicMethod) => set({ demosaicMethod }),
  setExportFormat: (exportFormat) => set({ exportFormat }),
  setExportQuality: (exportQuality) => set({ exportQuality }),
  setExportSaveTo: (exportSaveTo) => set({ exportSaveTo }),
  setExportOnExists: (exportOnExists) => set({ exportOnExists }),
});
