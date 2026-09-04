import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from '@/app/store';

describe('histogram mode state', () => {
  beforeEach(() => useAppStore.setState({ histogramSource: 'display', histogramChannel: 'rgb' }));
  it('updates source and channel', () => {
    useAppStore.getState().setHistogramSource('scene');
    useAppStore.getState().setHistogramChannel('ev');
    expect(useAppStore.getState().histogramSource).toBe('scene');
    expect(useAppStore.getState().histogramChannel).toBe('ev');
  });
});

describe('view state', () => {
  beforeEach(() => useAppStore.setState({ viewScale: 1, viewFitScale: 1, viewControls: null }));
  it('updates scale and fit', () => {
    useAppStore.getState().setViewScale(2.5);
    useAppStore.getState().setViewFitScale(0.2);
    expect(useAppStore.getState().viewScale).toBe(2.5);
    expect(useAppStore.getState().viewFitScale).toBe(0.2);
  });
  it('registers and clears view controls', () => {
    const controls = { zoomTo: () => {}, resetView: () => {}, panTo: () => {} };
    useAppStore.getState().setViewControls(controls);
    expect(useAppStore.getState().viewControls).toBe(controls);
    useAppStore.getState().setViewControls(null);
    expect(useAppStore.getState().viewControls).toBeNull();
  });
});
