import type { PanelId } from '@/renderer/grading/sections';
import type { HistogramChannel, Renderer } from '@/renderer';
import type { Slice, ViewControls } from './types';

export interface ViewSlice {
  openPanel: PanelId | null;
  histogramSource: 'scene' | 'display';
  histogramChannel: HistogramChannel;
  viewScale: number;
  viewFitScale: number;
  viewControls: ViewControls | null;
  viewPan: { x: number; y: number };
  viewContainerW: number;
  viewContainerH: number;
  /** The renderer of the displayed photo, published by OutputCanvas for export and the histogram HUD. */
  renderer: Renderer | null;
  setOpenPanel: (panel: PanelId | null) => void;
  togglePanel: (panel: PanelId) => void;
  setHistogramSource: (source: 'scene' | 'display') => void;
  setHistogramChannel: (channel: HistogramChannel) => void;
  setViewScale: (scale: number) => void;
  setViewFitScale: (fitScale: number) => void;
  setViewControls: (controls: ViewControls | null) => void;
  setViewPan: (pan: { x: number; y: number }) => void;
  setViewContainerSize: (w: number, h: number) => void;
  setRenderer: (renderer: Renderer | null) => void;
}

export const createViewSlice: Slice<ViewSlice> = (set) => ({
  openPanel: null,
  histogramSource: 'display',
  histogramChannel: 'rgb',
  viewScale: 1,
  viewFitScale: 1,
  viewControls: null,
  viewPan: { x: 0, y: 0 },
  viewContainerW: 0,
  viewContainerH: 0,
  renderer: null,
  setOpenPanel: (openPanel) => set({ openPanel }),
  togglePanel: (panel) => set((s) => ({ openPanel: s.openPanel === panel ? null : panel })),
  setHistogramSource: (histogramSource) => set({ histogramSource }),
  setHistogramChannel: (histogramChannel) => set({ histogramChannel }),
  setViewScale: (viewScale) => set({ viewScale }),
  setViewFitScale: (viewFitScale) => set({ viewFitScale }),
  setViewControls: (viewControls) => set({ viewControls }),
  setViewPan: (viewPan) => set({ viewPan }),
  setViewContainerSize: (viewContainerW, viewContainerH) => set({ viewContainerW, viewContainerH }),
  setRenderer: (renderer) => set({ renderer }),
});
