import { useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, PointerEvent } from 'react';
import { ChevronRight } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useAppStore } from '@/app/store';
import { CELL_GAP, COLUMN_STRIDE, maxColumns, shownIds } from '@/app/store/selection';
import { isDefaultPhotoEdit } from '@/app/photo-edit';
import { dismissBatch, exportMarks, useExportBatch } from '@/app/services/export-batch';
import { ExportFoot } from './ExportFoot';
import { FilterMenu } from './FilterMenu';
import './LibraryColumn.css';

export function LibraryColumn() {
  const shown = useAppStore(useShallow((s) => {
    const ids = new Set(shownIds(s));
    return s.files.filter((f) => ids.has(f.id));
  }));
  const hasFiles = useAppStore((s) => s.files.length > 0);
  const currentId = useAppStore((s) => s.currentId);
  const selectedIds = useAppStore((s) => s.selectedIds);
  const columnWidth = useAppStore((s) => s.columnWidth);
  const clickCell = useAppStore((s) => s.clickCell);
  const toggleSelectAll = useAppStore((s) => s.toggleSelectAll);
  const setColumnWidth = useAppStore((s) => s.setColumnWidth);
  const toggleColumn = useAppStore((s) => s.toggleColumn);
  const batch = useExportBatch((s) => s.batch);
  const marks = useMemo(() => exportMarks(batch), [batch]);
  // The width under the pointer while the edge is dragged; applied on release.
  const [preview, setPreview] = useState<number | null>(null);
  const dragRef = useRef<{ pointerId: number; columns: number } | null>(null);
  const rootRef = useRef<HTMLElement>(null);
  const currentRef = useRef<HTMLButtonElement>(null);
  const cellsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Optional: jsdom has no scrollIntoView.
    currentRef.current?.scrollIntoView?.({ block: 'nearest' });
    // Keyboard focus in the column follows the current cell; focus elsewhere stays put.
    if (cellsRef.current?.contains(document.activeElement)) currentRef.current?.focus({ preventScroll: true });
  }, [currentId, columnWidth]);

  // Re-clamp when the window shrinks, so the stage keeps its minimum width.
  useEffect(() => {
    const onResize = () => {
      const { columnWidth: columns, setColumnWidth: apply } = useAppStore.getState();
      apply(columns);
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  if (!hasFiles) return null;
  if (columnWidth === 0) {
    return (
      <button type="button" className="xv-column__tab" aria-label="Show photos" onClick={toggleColumn}>
        <ChevronRight size={10} />
      </button>
    );
  }

  const selected = new Set(selectedIds);
  const allShownSelected = shown.every((f) => selected.has(f.id));
  // Roving tabindex: the listbox is one tab stop, on the current cell (else the first).
  const tabStopId = shown.some((f) => f.id === currentId) ? currentId : shown[0]?.id;

  const onEdgeDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || dragRef.current) return;
    e.preventDefault();
    // Optional: jsdom has no pointer capture.
    e.currentTarget.setPointerCapture?.(e.pointerId);
    dragRef.current = { pointerId: e.pointerId, columns: columnWidth };
    setPreview(columnWidth);
  };
  const onEdgeMove = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId || !rootRef.current) return;
    const left = rootRef.current.getBoundingClientRect().left;
    const columns = Math.round((e.clientX - left - CELL_GAP) / COLUMN_STRIDE);
    drag.columns = Math.min(Math.max(columns, 0), maxColumns(window.innerWidth));
    setPreview(drag.columns);
  };
  const onEdgeEnd = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    dragRef.current = null;
    setPreview(null);
    setColumnWidth(drag.columns);
  };

  return (
    <aside
      ref={rootRef}
      className="xv-column"
      style={{ '--xv-columns': columnWidth } as CSSProperties}
      data-testid="library-column"
      data-dragging={preview !== null || undefined}
    >
      <FilterMenu />
      <div ref={cellsRef} className="xv-column__cells" role="listbox" aria-label="Photos" aria-multiselectable="true">
        {shown.map((f) => {
          const current = f.id === currentId;
          const mark = marks.get(f.id) ?? null;
          return (
            <button
              key={f.id}
              ref={current ? currentRef : undefined}
              type="button"
              role="option"
              className="xv-cell"
              data-testid="column-cell"
              aria-selected={selected.has(f.id)}
              data-current={current || undefined}
              data-rating={f.rating}
              aria-label={f.rating > 0 ? `${f.originalName}, rated ${f.rating}` : f.rating < 0 ? `${f.originalName}, rejected` : f.originalName}
              tabIndex={f.id === tabStopId ? 0 : -1}
              onClick={(e) => clickCell(f.id, { meta: e.metaKey || e.ctrlKey, shift: e.shiftKey })}
            >
              {f.thumbnailUrl && <img src={f.thumbnailUrl} alt="" draggable={false} />}
              {!isDefaultPhotoEdit(f.edit) && <span className="xv-cell__edited" />}
              {f.rating > 0 && <span className="xv-cell__stars" aria-hidden>{'★'.repeat(f.rating)}</span>}
              {f.rating < 0 && <span className="xv-cell__reject" aria-hidden>✕</span>}
              {mark && <span className="xv-cell__line" data-export={mark} aria-hidden />}
            </button>
          );
        })}
      </div>
      {/* Any click in the foot puts a done batch away; decided from the batch as rendered, so the
          click on Cancel that settles a batch to done doesn't also dismiss it. */}
      <div className="xv-column__foot" onClick={() => { if (batch?.status === 'done') dismissBatch(); }}>
        <button
          type="button"
          className="xv-column__select"
          aria-label={allShownSelected ? 'Select none' : 'Select all'}
          onClick={toggleSelectAll}
        >
          {allShownSelected ? 'None' : 'All'}
        </button>
        <ExportFoot />
      </div>
      <div
        className="xv-column__edge"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize photos"
        onPointerDown={onEdgeDown}
        onPointerMove={onEdgeMove}
        onPointerUp={onEdgeEnd}
        onPointerCancel={onEdgeEnd}
      />
    </aside>
  );
}
