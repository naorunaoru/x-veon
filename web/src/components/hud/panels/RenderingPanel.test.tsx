import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RenderingPanel } from './RenderingPanel';
import { ExposurePanel } from './ExposurePanel';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';
import { configFromPreset, configWithOverrides, TONESCALE_PRESETS } from '@/renderer/grading/opendrt-params';

function makeFile(overrides: Partial<QueuedFile> = {}): QueuedFile {
  return {
    id: 'a', file: null, name: 'a', originalName: 'a.raf', thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status: 'done', error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {}, ...overrides,
  };
}
const file = () => useAppStore.getState().files[0];
const selectLook = (value: string) => fireEvent.change(screen.getByRole('combobox', { name: 'Look' }), { target: { value } });

describe('RenderingPanel', () => {
  beforeEach(() => useAppStore.setState({ openPanel: 'advanced', files: [makeFile()], selectedFileId: 'a', displayHdr: false }));

  it('offers one look selector above Tone, Colour and collapsed Advanced controls', () => {
    render(<RenderingPanel />);
    expect(screen.getAllByRole('combobox')).toHaveLength(1);
    expect(screen.getByRole('combobox', { name: 'Look' })).toHaveDisplayValue('Default');
    expect(screen.getAllByRole('option', { name: 'Umbra' })).toHaveLength(1);
    expect(screen.queryByText('Base look')).not.toBeInTheDocument();
    expect(screen.queryByText('Tonescale preset')).not.toBeInTheDocument();
    ['Tone', 'Colour', 'Highlight warmth'].forEach((label) => expect(screen.getByText(label)).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /Advanced/ })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByText('peak_luminance')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }));
    expect(screen.getByText('peak_luminance')).toBeInTheDocument();
  });

  it('applies a complete look and can undo to the previous edited appearance', () => {
    const original = makeFile({ lookPreset: 'umbra', openDrtOverrides: { tn_con: 1.7, brl_r: 0.2 }, preProcessOverrides: { exposure: 1, wb_temp: 0.3, sharpen_amount: 0.5 } });
    useAppStore.setState({ files: [original] });
    render(<RenderingPanel />);
    expect(screen.getByRole('combobox', { name: 'Look' })).toHaveDisplayValue('Umbra · Modified');
    selectLook('marvelous');
    expect(file()).toMatchObject({ lookPreset: 'marvelous', openDrtOverrides: {}, preProcessOverrides: original.preProcessOverrides });
    expect(screen.getByRole('combobox', { name: 'Look' })).toHaveDisplayValue('Marvelous');
    expect(screen.getByRole('slider', { name: 'Contrast' })).toHaveAttribute('aria-valuenow', '1.5');
    fireEvent.click(screen.getByRole('button', { name: 'Undo look change' }));
    expect(file()).toMatchObject(original);
    expect(screen.getByRole('combobox', { name: 'Look' })).toHaveDisplayValue('Umbra · Modified');
  });

  it('opens legacy look-plus-tonescale edits without changing any rendering values', () => {
    const original = makeFile({ lookPreset: 'colorful', openDrtOverrides: { ...TONESCALE_PRESETS['aces-2'].overrides, cwp: 0.4 } });
    const before = configWithOverrides(configFromPreset(original.lookPreset), original.openDrtOverrides);
    useAppStore.setState({ files: [original] });
    render(<RenderingPanel />);
    expect(screen.getByRole('combobox', { name: 'Look' })).toHaveDisplayValue('Colorful · Modified');
    expect(configWithOverrides(configFromPreset(file().lookPreset), file().openDrtOverrides)).toEqual(before);
  });

  it('resets the whole Tone section to the selected look while preserving colour and corrections', () => {
    useAppStore.setState({ files: [makeFile({ lookPreset: 'aces-2', openDrtOverrides: { tn_con: 2, tn_lcon_pc: 0.2, tn_hcon_st: 3, cwp: 0.4 }, preProcessOverrides: { exposure: 1 } })] });
    render(<RenderingPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'Reset Tone' }));
    expect(file().openDrtOverrides).toEqual({ cwp: 0.4 });
    expect(file().preProcessOverrides).toEqual({ exposure: 1 });
    expect(screen.getByRole('slider', { name: 'Contrast' })).toHaveAttribute('aria-valuenow', '1.15');
    expect(screen.queryByRole('button', { name: 'Reset Tone' })).not.toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Look' })).toHaveDisplayValue('ACES-inspired · Modified');
  });

  it('resets all Colour controls, including warmth and advanced purity, without changing tone', () => {
    useAppStore.setState({ files: [makeFile({ openDrtOverrides: { tn_con: 1.6, cwp: 0.4, rs_rw: 0.8, ptl_enable: false, ptm_high_st: 0.7, hs_m: 0.3 } })] });
    render(<RenderingPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'Reset Colour' }));
    expect(file().openDrtOverrides).toEqual({ tn_con: 1.6 });
  });

  it('resets the selected look including advanced output adjustments and allows undo', () => {
    const original = makeFile({ lookPreset: 'umbra', openDrtOverrides: { tn_con: 1.6, cwp: 0.5, grey_boost: 0.2 }, preProcessOverrides: { wb_tint: 0.2, sharpen_amount: 0.5 } });
    useAppStore.setState({ files: [original] });
    render(<RenderingPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'Reset look' }));
    expect(file()).toMatchObject({ lookPreset: 'umbra', openDrtOverrides: {}, preProcessOverrides: original.preProcessOverrides });
    expect(screen.getByRole('combobox', { name: 'Look' })).toHaveDisplayValue('Umbra');
    fireEvent.click(screen.getByRole('button', { name: 'Undo look change' }));
    expect(file()).toMatchObject(original);
  });

  it('clears the Modified label when adjustments return to the look values', () => {
    render(<RenderingPanel />);
    const contrast = screen.getByRole('slider', { name: 'Contrast' });
    fireEvent.keyDown(contrast, { key: 'ArrowRight' });
    expect(screen.getByRole('combobox', { name: 'Look' })).toHaveDisplayValue('Default · Modified');
    fireEvent.keyDown(contrast, { key: 'ArrowLeft' });
    expect(screen.getByRole('combobox', { name: 'Look' })).toHaveDisplayValue('Default');
    expect(screen.queryByRole('button', { name: 'Reset Tone' })).not.toBeInTheDocument();
  });

  it('enables local contrast when adjusting a look or legacy edit with that module disabled', () => {
    useAppStore.setState({ files: [makeFile({ lookPreset: 'aces-2', openDrtOverrides: { tn_lcon_enable: false } })] });
    render(<RenderingPanel />);
    const slider = screen.getByRole('slider', { name: 'Local contrast' });
    expect(slider).toHaveAttribute('aria-valuenow', '0');
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    const cfg = configWithOverrides(configFromPreset(file().lookPreset), file().openDrtOverrides);
    expect(cfg.tn_lcon_enable).toBe(true);
    expect(cfg.tn_lcon).toBe(0.01);
  });

  it('leaves Exposure responsible only for exposure compensation and its reset', () => {
    useAppStore.setState({ files: [makeFile({ openDrtOverrides: { tn_con: 1.6 }, preProcessOverrides: { exposure: 1 } })] });
    render(<ExposurePanel />);
    expect(screen.getAllByRole('slider')).toHaveLength(1);
    expect(screen.getByRole('slider', { name: 'Exposure' })).toBeInTheDocument();
    expect(screen.queryByText('Local contrast')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Reset/ }));
    expect(file().preProcessOverrides).toEqual({});
    expect(file().openDrtOverrides).toEqual({ tn_con: 1.6 });
  });

  it('disables look selection when no photo is selected', () => {
    useAppStore.setState({ selectedFileId: null });
    render(<RenderingPanel />);
    expect(screen.getByRole('combobox', { name: 'Look' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Undo look change' })).not.toBeInTheDocument();
  });
});
