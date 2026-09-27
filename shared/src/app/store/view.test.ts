import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from '@/app/store';

describe('view pan + container state', () => {
  beforeEach(() => useAppStore.setState({ viewPan: { x: 0, y: 0 }, viewContainerW: 0, viewContainerH: 0 }));
  it('updates pan and container size', () => {
    useAppStore.getState().setViewPan({ x: -120, y: -40 });
    useAppStore.getState().setViewContainerSize(1280, 800);
    expect(useAppStore.getState().viewPan).toEqual({ x: -120, y: -40 });
    expect(useAppStore.getState().viewContainerW).toBe(1280);
    expect(useAppStore.getState().viewContainerH).toBe(800);
  });
  it('ViewControls now carries panTo', () => {
    const controls = { zoomTo: () => {}, resetView: () => {}, panTo: () => {} };
    useAppStore.getState().setViewControls(controls);
    expect(typeof useAppStore.getState().viewControls?.panTo).toBe('function');
  });
});
