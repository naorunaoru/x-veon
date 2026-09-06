import { act, fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
vi.mock('./panels/ExposurePanel', () => ({ ExposurePanel: () => <input aria-label="Exposure value" /> }));
vi.mock('./panels/RenderingPanel', () => ({ RenderingPanel: () => <input aria-label="Contrast value" /> }));
vi.mock('./panels/WhiteBalancePanel', () => ({ WhiteBalancePanel: () => <span>White balance controls</span> }));
vi.mock('./panels/DetailPanel', () => ({ DetailPanel: () => <span>Detail controls</span> }));
vi.mock('./panels/SettingsPanel', () => ({ SettingsPanel: () => <span>Settings controls</span> }));
import { PanelHost } from './PanelHost';
import { ToolRail } from './ToolRail';
import { useAppStore } from '@/app/store';
const scrollTo = vi.fn(function (this: HTMLElement, options: ScrollToOptions) {
  this.scrollTop = options.top ?? this.scrollTop;
});
beforeEach(() => {
  scrollTo.mockClear();
  // jsdom has no scrolling implementation; record the requested behavior while
  // keeping the existing geometry/navigation tests synchronous.
  Object.defineProperty(HTMLElement.prototype, 'scrollTo', { configurable: true, value: scrollTo });
  useAppStore.setState({ files: [], openPanel: null, panelNavigationRevision: 0 });
});
it('keeps all adjustments together, scrolls on repeated navigation, and tracks manual scrolling', () => {
  const { container } = render(<><PanelHost /><ToolRail /></>);
  const body = container.querySelector('.xv-adjustments__body') as HTMLElement;
  const sections = container.querySelectorAll('[data-adjustment]');
  sections.forEach((section, i) => Object.defineProperty(section, 'offsetTop', { value: i * 200 }));
  fireEvent.click(screen.getByRole('button', { name: 'Rendering' }));
  expect(body.scrollTop).toBe(200);
  expect(screen.getByLabelText('Exposure value')).toBeVisible();
  expect(screen.getByLabelText('Contrast value')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Rendering' }));
  expect(screen.getByLabelText('Contrast value')).toBeVisible();
  body.scrollTop = 430; fireEvent.scroll(body);
  expect(screen.getByRole('button', { name: 'White balance' })).toHaveAttribute('aria-pressed', 'true');
  expect(body.scrollTop).toBe(430);
  act(() => useAppStore.setState({ selectedFileId: 'other' }));
  expect(body.scrollTop).toBe(430);
  fireEvent.click(screen.getByRole('button', { name: 'Exposure' }));
  expect(body.scrollTop).toBe(0);
  fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
  expect(screen.getByText('Settings controls')).toBeVisible();
  expect(screen.getByLabelText('Exposure value')).not.toBeVisible();
  expect(screen.queryByRole('button', { name: 'Scopes' })).toBeNull();
});

it('allows centering Detail without painting the transparent scroll tail', () => {
  const { container } = render(<><PanelHost /><ToolRail /></>);
  const panel = container.querySelector('.xv-adjustments') as HTMLElement;
  const body = container.querySelector('.xv-adjustments__body') as HTMLElement;
  const content = container.querySelector('.xv-adjustments__content') as HTMLElement;
  const detail = container.querySelector('[data-adjustment="detail"]') as HTMLElement;
  const viewport = vi.spyOn(window, 'innerHeight', 'get').mockReturnValue(1000);
  Object.defineProperties(body, { clientHeight: { value: 900 }, offsetTop: { value: 58 }, scrollHeight: { get: () => 1400 + parseFloat(body.style.getPropertyValue('--scroll-tail') || '0') } });
  body.getBoundingClientRect = () => ({ top: 70 }) as DOMRect;
  Object.defineProperty(content, 'offsetHeight', { value: 1400 });
  Object.defineProperties(detail, { offsetTop: { value: 1240 }, offsetHeight: { value: 160 } });
  let scroll = 0;
  Object.defineProperty(body, 'scrollTop', {
    get: () => scroll,
    set: (value: number) => { scroll = Math.max(0, Math.min(value, body.scrollHeight - body.clientHeight)); },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Detail' }));
  // Detail spans y=420..580, centered in the 1000px viewport.
  expect(body.scrollTop).toBe(890);
  expect(70 + detail.offsetTop - body.scrollTop + detail.offsetHeight / 2).toBe(500);
  expect(body.style.getPropertyValue('--visible-content-height')).toBe('510px');
  expect(panel.style.getPropertyValue('--surface-height')).toBe('569px');
  fireEvent.scroll(body);
  expect(screen.getByRole('button', { name: 'Detail' })).toHaveAttribute('aria-pressed', 'true');
  body.scrollTop = 0;
  fireEvent.scroll(body);
  expect(body.style.getPropertyValue('--visible-content-height')).toBe('900px');
  viewport.mockRestore();
});

it('keeps wheel input in the panel column across the full viewport, even over the photo below the surface', () => {
  const { container } = render(<><div data-testid="photo" /><PanelHost /><ToolRail /></>);
  const panel = container.querySelector('.xv-adjustments') as HTMLElement;
  const body = container.querySelector('.xv-adjustments__body') as HTMLElement;
  const photo = screen.getByTestId('photo');
  const photoWheel = vi.fn();
  photo.addEventListener('wheel', photoWheel);
  panel.getBoundingClientRect = () => ({ left: 800, right: 1188, top: 12, bottom: 708 }) as DOMRect;
  fireEvent.click(screen.getByRole('button', { name: 'Detail' }));
  body.scrollTop = 1000;
  const wheel = new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: 1000, clientY: 719, deltaY: -120 });
  fireEvent(photo, wheel);
  expect(body.scrollTop).toBe(880);
  expect(wheel.defaultPrevented).toBe(true);
  expect(photoWheel).not.toHaveBeenCalled();
  fireEvent.wheel(photo, { clientX: 1000, clientY: 1, deltaY: -3, deltaMode: 1 });
  expect(body.scrollTop).toBe(832);
  fireEvent.wheel(photo, { clientX: 400, clientY: 500, deltaY: -120 });
  expect(body.scrollTop).toBe(832);
  expect(photoWheel).toHaveBeenCalledTimes(1);
  fireEvent.wheel(photo, { clientX: 1000, clientY: 719, deltaY: -120, ctrlKey: true });
  expect(photoWheel).toHaveBeenCalledTimes(2);
  fireEvent.click(screen.getByRole('button', { name: 'Close panel' }));
  fireEvent.wheel(photo, { clientX: 1000, clientY: 719, deltaY: -120 });
  expect(body.scrollTop).toBe(832);
  expect(photoWheel).toHaveBeenCalledTimes(3);
});

it('centers short groups including Exposure, and top-aligns groups taller than the viewport', () => {
  const { container } = render(<><PanelHost /><ToolRail /></>);
  const panel = container.querySelector('.xv-adjustments') as HTMLElement;
  const body = container.querySelector('.xv-adjustments__body') as HTMLElement;
  const content = container.querySelector('.xv-adjustments__content') as HTMLElement;
  const groups = Array.from(container.querySelectorAll('[data-adjustment]')) as HTMLElement[];
  const viewport = vi.spyOn(window, 'innerHeight', 'get').mockReturnValue(1000);
  const head = () => parseFloat(body.style.getPropertyValue('--scroll-head') || '0');
  const tail = () => parseFloat(body.style.getPropertyValue('--scroll-tail') || '0');
  Object.defineProperties(body, { clientHeight: { value: 900 }, offsetTop: { value: 58 }, scrollHeight: { get: () => head() + 1600 + tail() } });
  body.getBoundingClientRect = () => ({ top: 70 }) as DOMRect;
  Object.defineProperties(content, { offsetHeight: { value: 1600 }, offsetTop: { get: head } });
  const offsets = [0, 100, 1200, 1440], heights = [100, 1100, 240, 160];
  groups.forEach((group, i) => Object.defineProperties(group, { offsetTop: { get: () => head() + offsets[i] }, offsetHeight: { value: heights[i] } }));
  let scroll = 0;
  Object.defineProperty(body, 'scrollTop', {
    get: () => scroll,
    set: (value: number) => { scroll = Math.max(0, Math.min(value, body.scrollHeight - body.clientHeight)); },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Exposure' }));
  expect(70 + groups[0].offsetTop - scroll + 50).toBe(500);
  expect(panel.style.getPropertyValue('--surface-top')).toBe('380px');
  fireEvent.click(screen.getByRole('button', { name: 'White balance' }));
  expect(70 + groups[2].offsetTop - scroll + 120).toBe(500);
  expect(panel.style.getPropertyValue('--surface-top')).toBe('0px');
  fireEvent.scroll(body);
  expect(screen.getByRole('button', { name: 'White balance' })).toHaveAttribute('aria-pressed', 'true');
  fireEvent.click(screen.getByRole('button', { name: 'Rendering' }));
  expect(70 + groups[1].offsetTop - scroll).toBe(70);
  viewport.mockRestore();
});

it('animates toolbar navigation only when the panel was already open', () => {
  const { container } = render(<><PanelHost /><ToolRail /></>);
  const body = container.querySelector('.xv-adjustments__body') as HTMLElement;
  container.querySelectorAll('[data-adjustment]').forEach((section, i) => Object.defineProperty(section, 'offsetTop', { value: i * 200 }));
  fireEvent.click(screen.getByRole('button', { name: 'Rendering' }));
  expect(body.scrollTop).toBe(200);
  expect(scrollTo).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Detail' }));
  expect(scrollTo).toHaveBeenLastCalledWith({ top: 600, behavior: 'smooth' });
  fireEvent.scroll(body);
  act(() => useAppStore.setState({ selectedFileId: 'another' }));
  expect(scrollTo).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Exposure' }));
  expect(scrollTo).toHaveBeenLastCalledWith({ top: 0, behavior: 'smooth' });
  fireEvent.click(screen.getByRole('button', { name: 'Close panel' }));
  fireEvent.click(screen.getByRole('button', { name: 'White balance' }));
  expect(body.scrollTop).toBe(400);
  expect(scrollTo).toHaveBeenCalledTimes(2);
});

it('respects reduced motion for toolbar navigation', () => {
  vi.stubGlobal('matchMedia', vi.fn(() => ({ matches: true })));
  try {
    const { container } = render(<><PanelHost /><ToolRail /></>);
    const body = container.querySelector('.xv-adjustments__body') as HTMLElement;
    container.querySelectorAll('[data-adjustment]').forEach((section, i) => Object.defineProperty(section, 'offsetTop', { value: i * 200 }));
    fireEvent.click(screen.getByRole('button', { name: 'Exposure' }));
    fireEvent.click(screen.getByRole('button', { name: 'Detail' }));
    expect(body.scrollTop).toBe(600);
    expect(scrollTo).not.toHaveBeenCalled();
  } finally {
    vi.unstubAllGlobals();
  }
});
