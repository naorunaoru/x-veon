import { useLayoutEffect, useRef } from 'react';
import { X } from 'lucide-react';
import { useAppStore } from '@/app/store';
import { ExposurePanel } from './panels/ExposurePanel';
import { WhiteBalancePanel } from './panels/WhiteBalancePanel';
import { SettingsPanel } from './panels/SettingsPanel';
import { RenderingPanel } from './panels/RenderingPanel';
import { DetailPanel } from './panels/DetailPanel';
import type { PanelId } from '@/renderer/grading/sections';
import './AdjustmentsPanel.css';

const ADJUSTMENT_GROUPS = ['exposure', 'advanced', 'whiteBalance', 'detail'] as const;
const isAdjustment = (id: PanelId | null) => ADJUSTMENT_GROUPS.some((group) => group === id);

export function PanelHost({ settingsOnly = false }: { settingsOnly?: boolean }) {
  const viewOnly = useAppStore(s => s.files.find(f => f.id === s.selectedFileId)?.editing === 'view-only');
  const openPanel = useAppStore((s) => s.openPanel);
  const revision = useAppStore((s) => s.panelNavigationRevision);
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const panel = useRef<HTMLDivElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const lastNavigation = useRef<{ revision: number; adjusting: boolean } | null>(null);
  const adjusting = !settingsOnly && isAdjustment(openPanel);

  function updateSurface() {
    if (!body.current || !content.current || !panel.current) return;
    const start = Math.max(0, content.current.offsetTop - body.current.scrollTop);
    const end = content.current.offsetTop + content.current.offsetHeight - body.current.scrollTop;
    const visibleHeight = Math.max(0, Math.min(body.current.clientHeight, end));
    body.current.style.setProperty('--visible-content-start', `${start}px`);
    body.current.style.setProperty('--visible-content-height', `${visibleHeight}px`);
    panel.current.style.setProperty('--surface-top', `${start}px`);
    panel.current.style.setProperty('--surface-height', `${body.current.offsetTop + visibleHeight + 1 - start}px`);
  }

  function updateScrollRange() {
    const container = body.current;
    const first = content.current?.firstElementChild as HTMLElement | null;
    const last = content.current?.lastElementChild as HTMLElement | null;
    if (!container || !first || !last) return;
    // Allow the first and last groups to reach the viewport center. These
    // transparent margins are excluded from the painted/click area;
    // wheel input still belongs to the panel throughout this viewport column.
    const center = window.innerHeight / 2 - container.getBoundingClientRect().top;
    const head = first.offsetHeight > 0 ? Math.max(0, center - first.offsetHeight / 2) : 0;
    const tail = Math.max(0, container.clientHeight - center - last.offsetHeight / 2);
    container.style.setProperty('--scroll-head', `${head}px`);
    container.style.setProperty('--scroll-tail', `${tail}px`);
    updateSurface();
  }

  useLayoutEffect(() => {
    if (!adjusting || !body.current || !content.current) return;
    updateScrollRange();
    const observer = new ResizeObserver(updateScrollRange);
    observer.observe(body.current);
    observer.observe(content.current);
    window.addEventListener('resize', updateScrollRange);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updateScrollRange);
    };
  }, [adjusting]);

  useLayoutEffect(() => {
    if (!adjusting) return;
    const onWheel = (event: WheelEvent) => {
      const shell = panel.current;
      const container = body.current;
      if (!shell || !container || event.defaultPrevented || event.ctrlKey || event.metaKey) return;
      if (shell.closest('[data-chrome-hidden="true"]')) return;
      if (document.querySelector('[role="dialog"], [role="alertdialog"]')) return;
      const rect = shell.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX >= rect.right) return;
      if (event.clientY < rect.top || event.clientY >= rect.bottom) return;
      // Capture before the photo's wheel listener, including below the shortened
      // glass surface. The column stops short of the top bar and the bottom HUD row,
      // so the filmstrip and zoom controls keep their own wheel handling.
      event.preventDefault();
      event.stopPropagation();
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? container.clientHeight : 1;
      container.scrollTop += event.deltaY * unit;
    };
    window.addEventListener('wheel', onWheel, { capture: true, passive: false });
    return () => window.removeEventListener('wheel', onWheel, { capture: true });
  }, [adjusting]);

  // Only explicit navigation scrolls the panel. Scroll tracking and photo edits
  // update the highlighted group without initiating another scroll.
  useLayoutEffect(() => {
    const previous = lastNavigation.current;
    lastNavigation.current = { revision, adjusting };
    if (!adjusting || !body.current) return;
    updateScrollRange();
    const target = body.current.querySelector<HTMLElement>(`[data-adjustment="${useAppStore.getState().openPanel}"]`);
    if (target) {
      const center = window.innerHeight / 2 - body.current.getBoundingClientRect().top;
      const top = target.offsetHeight > 0 && target.offsetHeight < body.current.clientHeight
        ? target.offsetTop + target.offsetHeight / 2 - center
        : target.offsetTop;
      const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
      if (previous?.adjusting && previous.revision !== revision && !reducedMotion) {
        body.current.scrollTo({ top, behavior: 'smooth' });
      } else {
        body.current.scrollTop = top;
      }
    }
    updateSurface();
  }, [revision, adjusting]);

  function trackSection() {
    const container = body.current;
    if (!container || !isAdjustment(useAppStore.getState().openPanel)) return;
    updateSurface();
    const sections = Array.from(container.querySelectorAll<HTMLElement>('[data-adjustment]'));
    const atEnd = container.clientHeight > 0 && container.scrollTop > 0 && container.scrollTop >= container.scrollHeight - container.clientHeight - 1;
    const anchor = container.clientHeight > 0 ? window.innerHeight / 2 - container.getBoundingClientRect().top : 24;
    const active = atEnd ? sections.at(-1) : sections.filter((section) => section.offsetTop <= container.scrollTop + anchor).at(-1) ?? sections[0];
    if (active) useAppStore.getState().setActiveAdjustment(active.dataset.adjustment as PanelId);
  }

  return <>
    <div ref={panel} className="xv-panel xv-adjustments" hidden={!adjusting} aria-label="Adjustments">
      <div className="xv-adjustments__surface xv-glass-heavy" aria-hidden="true" />
      <button className="xv-panel__icon xv-adjustments__close" aria-label="Close panel" onClick={() => setOpenPanel(null)}><X size={14} /></button>
      <div className="xv-adjustments__body" ref={body} onScroll={trackSection}>
        <div ref={content} className="xv-adjustments__content" inert={viewOnly} aria-disabled={viewOnly}>
          <section data-adjustment="exposure" aria-label="Exposure"><ExposurePanel embedded /></section>
          <section data-adjustment="advanced" aria-label="Rendering"><RenderingPanel embedded /></section>
          <section data-adjustment="whiteBalance" aria-label="White balance"><WhiteBalancePanel embedded /></section>
          <section data-adjustment="detail" aria-label="Detail"><DetailPanel embedded /></section>
        </div>
      </div>
    </div>
    {openPanel === 'settings' && <SettingsPanel />}
  </>;
}
