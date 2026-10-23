import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { LensPanel } from './LensPanel';
import { useAppStore, type QueuedFile } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto } from '@/test/fake-host';
import { deserializeResultMeta, type ProcessingMetadata, type SerializableResultMeta } from '@/lib/types';
import { DEFAULT_LENS_EDIT } from '@/lib/lens/edit';
import { XT3 } from '@/lib/lens/fuji-fixtures';
import type { LensProfile } from '@/lib/lensfun-types';
import { listLenses } from '@/app/lens/lensfun';
import { ensureLensPick } from '@/app/services/lens-picks';

vi.mock('@/app/lens/lensfun', () => ({ listLenses: vi.fn(async () => []) }));
vi.mock('@/app/services/lens-picks', () => ({ ensureLensPick: vi.fn() }));

const XT3_RESULT: SerializableResultMeta = {
  exportData: {
    width: 6384, height: 4182, xyzToCam: null, wbCoeffs: [1, 1, 1],
    camToXyz: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], orientation: 'Normal',
  },
  metadata: {
    lensModel: 'XF35mmF2 R WR', focalLength: 35, fNumber: 3.6, width: 6384, height: 4182,
    fujiCorrection: { ...XT3, centre: null },
  } as ProcessingMetadata,
};

/** A processed X-T3 photo with its lens matched. */
function xt3(over: Partial<QueuedFile> = {}): QueuedFile {
  return {
    ...fromLibraryPhoto(fakePhoto()),
    status: 'done',
    lensMatched: true,
    metadata: { camera: 'Fujifilm X-T3', lensModel: 'XF35mmF2 R WR', focalLength: 35, fNumber: 3.6 },
    result: deserializeResultMeta(XT3_RESULT),
    ...over,
  };
}

const edit = () => useAppStore.getState().files[0].edit;
const disabledSliders = () => document.querySelectorAll('.xv-slider.is-disabled').length;
const sliders = () => screen.getAllByRole('slider').map((thumb) => Number(thumb.getAttribute('aria-valuenow')));

/** Mounts the panel and lets the lens list settle. */
async function open() {
  const view = render(<LensPanel embedded />);
  await act(async () => {});
  return view;
}

beforeEach(() => {
  vi.clearAllMocks();
  useAppStore.setState({ files: [xt3()], currentId: 'a', lensPicks: {} });
});

describe('LensPanel', () => {
  it('shows the camera source and disables the sliders in Auto', async () => {
    await open();
    expect(screen.getByText('Camera · XF35mmF2 R WR')).toBeInTheDocument();
    expect(disabledSliders()).toBe(5);
    expect(screen.getByRole('combobox')).toHaveValue('auto');
  });

  it('shows the profile\'s effect at the corner on the disabled sliders', async () => {
    await open();
    const [distortion, caRed, caBlue, vignetting, midpoint] = sliders();
    expect(distortion).toBeGreaterThan(6);
    expect(distortion).toBeLessThan(8);
    expect(caRed).toBeCloseTo(-4, 0);
    expect(caBlue).toBeCloseTo(5, 0);
    expect(vignetting).toBeCloseTo(0, 5);
    expect(midpoint).toBe(0.5);
  });

  it('shows the profile as no edit on the sliders, and a manual value as one', async () => {
    await open();
    expect(document.querySelectorAll('.xv-slider.is-modified')).toHaveLength(0);
    expect(document.querySelectorAll('.xv-slider__delta')).toHaveLength(0);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'manual' } });
    expect(document.querySelectorAll('.xv-slider.is-modified')).toHaveLength(0);
    act(() => useAppStore.getState().setFileLens('a', { manual: { ...DEFAULT_LENS_EDIT.manual, caBlue: 2 } }));
    const modified = document.querySelectorAll('.xv-slider.is-modified');
    expect(modified).toHaveLength(1);
    expect(modified[0]).toHaveTextContent('Blue/yellow');
  });

  it('switching to Manual enables the sliders and marks the section modified', async () => {
    await open();
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'manual' } });
    expect(edit().lens?.source).toBe('manual');
    expect(disabledSliders()).toBe(0);
    // the picker's option reads "Manual" too
    expect(screen.getByText('Manual', { selector: '.xv-readout' })).toBeInTheDocument();
    expect(screen.getByLabelText('Reset to preset')).toBeInTheDocument();
  });

  it('shows the manual values on the live sliders and writes a change beside the others', async () => {
    useAppStore.setState({
      files: [xt3({ edit: { ...edit(), lens: { ...DEFAULT_LENS_EDIT, source: 'manual', manual: { ...DEFAULT_LENS_EDIT.manual, caBlue: 2 } } } })],
    });
    await open();
    expect(sliders()).toEqual([0, 0, 2, 0, 0.5]);
    fireEvent.keyDown(screen.getAllByRole('slider', { name: 'Amount' })[0], { key: 'ArrowRight' });
    expect(edit().lens?.manual).toEqual({ ...DEFAULT_LENS_EDIT.manual, distortion: 0.1, caBlue: 2 });
  });

  it('toggles write the edit', async () => {
    await open();
    fireEvent.click(screen.getByRole('switch', { name: 'Vignetting' }));
    expect(edit().lens?.vignetting).toBe(false);
    fireEvent.click(screen.getByRole('switch', { name: 'Distortion' }));
    fireEvent.click(screen.getByRole('switch', { name: 'Chromatic aberration' }));
    expect(edit().lens).toMatchObject({ distortion: false, ca: false, vignetting: false });
    expect(screen.getAllByRole('switch').map((s) => s.getAttribute('aria-checked'))).toEqual(['false', 'false', 'false']);
  });

  it('shows Matching lens while the match is in flight', async () => {
    useAppStore.setState((s) => ({ files: s.files.map((f) => ({ ...f, lensMatched: false })) }));
    await open();
    expect(screen.getByText('Matching lens')).toBeInTheDocument();
    expect(disabledSliders()).toBe(5);
  });

  describe('on a photo without camera tables, whose lensfun match is in flight', () => {
    const plain = (over: Partial<QueuedFile>) => xt3({
      result: deserializeResultMeta({ ...XT3_RESULT, metadata: { ...XT3_RESULT.metadata, fujiCorrection: undefined } }),
      lensProfile: null,
      ...over,
    });

    it('keeps the sliders disabled until the match lands', async () => {
      useAppStore.setState({ files: [plain({ lensMatched: false })] });
      await open();
      expect(screen.getByText('Matching lens')).toBeInTheDocument();
      expect(disabledSliders()).toBe(5);
    });

    it('makes them live once the match finds no profile', async () => {
      useAppStore.setState({ files: [plain({ lensMatched: true })] });
      await open();
      expect(screen.getByText('No profile for XF35mmF2 R WR')).toBeInTheDocument();
      expect(disabledSliders()).toBe(0);
    });

    it('leaves a chosen Manual live while the match is in flight', async () => {
      useAppStore.setState({ files: [plain({ lensMatched: false, edit: { ...edit(), lens: { ...DEFAULT_LENS_EDIT, source: 'manual' } } })] });
      await open();
      expect(screen.getByText('Manual', { selector: '.xv-readout' })).toBeInTheDocument();
      expect(disabledSliders()).toBe(0);
    });
  });

  it('reset returns the photo to the defaults', async () => {
    useAppStore.setState({ files: [xt3({ edit: { ...edit(), lens: { ...DEFAULT_LENS_EDIT, source: 'manual', ca: false } } })] });
    await open();
    fireEvent.click(screen.getByLabelText('Reset to preset'));
    expect(edit().lens).toBeUndefined();
    expect(screen.getByRole('combobox')).toHaveValue('auto');
    expect(screen.queryByLabelText('Reset to preset')).toBeNull();
  });

  it('waits for the decode with everything disabled', async () => {
    useAppStore.setState({ files: [xt3({ result: null })] });
    await open();
    expect(screen.getByText('Waiting for decode')).toBeInTheDocument();
    expect(screen.getByRole('combobox')).toBeDisabled();
    screen.getAllByRole('switch').forEach((toggle) => expect(toggle).toBeDisabled());
    expect(disabledSliders()).toBe(5);
  });

  it('names a hand-picked lens that is still loading, never the automatic match', async () => {
    const automatic: LensProfile = {
      lensModel: 'Automatic Match', mount: 'Fujifilm X', cropfactor: 1.5, calibCropfactor: 1.5, calibAspectRatio: 1.5,
      distortion: [], tca: [], vignetting: [],
    };
    const source = { lensfun: { file: 'misc.json', model: 'Picked Lens' } };
    useAppStore.setState({ files: [xt3({ lensProfile: automatic, edit: { ...edit(), lens: { ...DEFAULT_LENS_EDIT, source } } })] });
    await open();
    expect(screen.getByText('Lensfun · Picked Lens')).toBeInTheDocument();
  });

  it('renders for a photo without metadata and asks for no lens list', async () => {
    useAppStore.setState({ files: [xt3({ metadata: null })] });
    await open();
    expect(screen.getByText('Camera')).toBeInTheDocument();
    expect(listLenses).not.toHaveBeenCalled();
  });

  it('renders without a selected photo', async () => {
    useAppStore.setState({ files: [], currentId: null });
    await open();
    expect(screen.getByText('Waiting for decode')).toBeInTheDocument();
    expect(screen.getByRole('combobox')).toBeDisabled();
  });
});

describe('LensPanel lens list', () => {
  const choices = [
    { file: 'mil-fujifilm.json', make: 'Fujifilm', model: 'XF27mmF2.8 R WR' },
    { file: 'misc.json', make: 'Schneider', model: 'Schneider 28mm Digitar f/2.8' },
  ];
  const withCamera = (camera: string) => xt3({ metadata: { camera, lensModel: 'XF35mmF2 R WR', focalLength: 35, fNumber: 3.6 } });

  it('lists the camera\'s lenses under Lensfun and picks one by hand', async () => {
    vi.mocked(listLenses).mockResolvedValueOnce(choices);
    useAppStore.setState({ files: [withCamera('Fujifilm X-T3 pick')] });
    await open();
    expect(listLenses).toHaveBeenCalledWith('Fujifilm X-T3 pick');
    const group = document.querySelector('optgroup[label="Lensfun"]')!;
    expect([...group.querySelectorAll('option')].map((o) => [o.value, o.textContent])).toEqual([
      ['lensfun:mil-fujifilm.json#XF27mmF2.8 R WR', 'Fujifilm XF27mmF2.8 R WR'],
      ['lensfun:misc.json#Schneider 28mm Digitar f/2.8', 'Schneider 28mm Digitar f/2.8'],
    ]);

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'lensfun:mil-fujifilm.json#XF27mmF2.8 R WR' } });
    expect(edit().lens?.source).toEqual({ lensfun: { file: 'mil-fujifilm.json', model: 'XF27mmF2.8 R WR' } });
    expect(ensureLensPick).toHaveBeenCalledWith('a', { file: 'mil-fujifilm.json', model: 'XF27mmF2.8 R WR' });
    expect(screen.getByRole('combobox')).toHaveValue('lensfun:mil-fujifilm.json#XF27mmF2.8 R WR');
    expect(screen.getByText('Lensfun · XF27mmF2.8 R WR')).toBeInTheDocument();
    expect(disabledSliders()).toBe(5);
  });

  it('going back to Auto or Manual does not load a lens', async () => {
    useAppStore.setState({ files: [withCamera('Fujifilm X-T3 auto')] });
    await open();
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'manual' } });
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'auto' } });
    expect(edit().lens?.source).toBe('auto');
    expect(ensureLensPick).not.toHaveBeenCalled();
  });

  it('keeps a hand-picked lens selected before the list arrives', async () => {
    const source = { lensfun: { file: 'misc.json', model: 'Schneider 28mm Digitar f/2.8' } };
    useAppStore.setState({
      files: [xt3({ edit: { ...edit(), lens: { ...DEFAULT_LENS_EDIT, source } } })],
    });
    await open();
    expect(screen.getByRole('combobox')).toHaveValue('lensfun:misc.json#Schneider 28mm Digitar f/2.8');
  });

  it('fetches the list once per camera, however often the panel is reopened', async () => {
    useAppStore.setState({ files: [withCamera('Fujifilm X-T3 cache')] });
    const first = await open();
    first.unmount();
    await open();
    expect(listLenses).toHaveBeenCalledTimes(1);
  });

  it('shows no lenses when the list fails, and tries again on the next open', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.mocked(listLenses).mockRejectedValueOnce(new Error('offline'));
    useAppStore.setState({ files: [withCamera('Fujifilm X-T3 offline')] });
    const first = await open();
    expect(document.querySelectorAll('optgroup[label="Lensfun"] option')).toHaveLength(0);
    expect(warn).toHaveBeenCalled();
    first.unmount();
    await open();
    expect(listLenses).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });
});
