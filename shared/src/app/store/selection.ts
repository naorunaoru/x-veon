import * as reducer from '@/app/selection';
import type { SelectionOrder, SelectionState } from '@/app/selection';
import { matchesFilter } from '@/app/rating';
import type { RatingFilter } from '@/app/rating';
import type { AppState, Slice } from './types';

export const CELL_WIDTH = 112, CELL_GAP = 6, COLUMN_STRIDE = 118, STAGE_MIN_WIDTH = 480;

export interface SelectionSlice {
  selectedIds: string[];
  selectionAnchor: string | null;
  /** Columns; 0 = collapsed. Persisted. */
  columnWidth: number;
  /** The last non-zero width, for reopening; session only. */
  lastColumnWidth: number;
  /** Which ratings the column shows; session only. */
  filter: RatingFilter;
  /** Prunes the selection and anchor to the shown photos; a hidden current moves to the first shown one. */
  setFilter: (filter: RatingFilter) => void;
  clickCell: (id: string, mods: { meta: boolean; shift: boolean }) => void;
  moveCurrent: (delta: number, extend: boolean) => void;
  /** ⌘A: selects, never clears. */
  selectAllShown: () => void;
  /** The foot button. */
  toggleSelectAll: () => void;
  clearSelection: () => void;
  /** Clamps to [0, maxColumns(window.innerWidth)]. */
  setColumnWidth: (columns: number) => void;
  /** 0 ↔ lastColumnWidth. */
  toggleColumn: () => void;
}

/** The ids the column lists: the files whose rating matches the filter, in folder order. */
export function shownIds(state: Pick<AppState, 'files' | 'filter'>): string[] {
  return state.files.filter((f) => matchesFilter(f.rating, state.filter)).map((f) => f.id);
}

/** How many columns fit beside a stage of at least STAGE_MIN_WIDTH. */
export function maxColumns(windowWidth: number): number {
  if (!Number.isFinite(windowWidth)) return 0;
  return Math.max(0, Math.floor((windowWidth - STAGE_MIN_WIDTH - CELL_GAP) / COLUMN_STRIDE));
}

type ColumnWidths = Pick<SelectionSlice, 'columnWidth' | 'lastColumnWidth'>;

/** `columns` clamped to the window, recorded as the width to reopen to when it is not 0; `current` when it is not a number. */
export function columnWidthPatch(columns: number, current: ColumnWidths): ColumnWidths {
  const { lastColumnWidth } = current;
  if (!Number.isFinite(columns)) return { columnWidth: current.columnWidth, lastColumnWidth };
  const columnWidth = Math.min(Math.max(columns, 0), maxColumns(window.innerWidth));
  return { columnWidth, lastColumnWidth: columnWidth > 0 ? columnWidth : lastColumnWidth };
}

export const createSelectionSlice: Slice<SelectionSlice> = (set, get) => {
  /** Runs a reducer over the store's selection; sets nothing when it returns its input. */
  const apply = (step: (s: SelectionState, o: SelectionOrder) => SelectionState) => {
    const state = get();
    const before: SelectionState = { currentId: state.currentId, selectedIds: state.selectedIds, anchorId: state.selectionAnchor };
    const after = step(before, { order: state.files.map((f) => f.id), shown: new Set(shownIds(state)) });
    if (after !== before) set({ currentId: after.currentId, selectedIds: after.selectedIds, selectionAnchor: after.anchorId });
  };
  return {
    selectedIds: [],
    selectionAnchor: null,
    columnWidth: 1,
    lastColumnWidth: 1,
    filter: 'all',
    setFilter: (filter) => {
      const { files, currentId, selectedIds, selectionAnchor } = get();
      const list = shownIds({ files, filter });
      const shown = new Set(list);
      const pruned = reducer.prune({ currentId, selectedIds, anchorId: selectionAnchor }, shown);
      set({
        filter, selectedIds: pruned.selectedIds, selectionAnchor: pruned.anchorId,
        currentId: currentId !== null && shown.has(currentId) ? currentId : list[0] ?? null,
      });
    },
    clickCell: (id, mods) => apply((s, o) => reducer.clickSelect(s, o, id, mods)),
    moveCurrent: (delta, extend) => apply((s, o) => reducer.moveCurrent(s, o, delta, extend)),
    selectAllShown: () => apply(reducer.selectAllShown),
    toggleSelectAll: () => apply(reducer.toggleSelectAll),
    clearSelection: () => apply(reducer.clearSelection),
    setColumnWidth: (columns) => {
      const { columnWidth, lastColumnWidth } = get();
      const next = columnWidthPatch(columns, { columnWidth, lastColumnWidth });
      if (next.columnWidth !== columnWidth || next.lastColumnWidth !== lastColumnWidth) set(next);
    },
    toggleColumn: () => {
      const { columnWidth, lastColumnWidth, setColumnWidth } = get();
      setColumnWidth(columnWidth > 0 ? 0 : lastColumnWidth);
    },
  };
};
