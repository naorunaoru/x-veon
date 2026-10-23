export interface SelectionState { currentId: string | null; selectedIds: string[]; anchorId: string | null }
/** `order` is the folder order of every file; `shown` holds the ids the column lists. */
export interface SelectionOrder { order: string[]; shown: ReadonlySet<string> }

function shownIds(o: SelectionOrder): string[] {
  return o.order.filter((id) => o.shown.has(id));
}

/** The shown ids between `from` and `to` in `order`, inclusive; both must be in `order`. */
function range(o: SelectionOrder, from: string, to: string): string[] {
  const i = o.order.indexOf(from);
  const j = o.order.indexOf(to);
  return o.order.slice(Math.min(i, j), Math.max(i, j) + 1).filter((id) => o.shown.has(id));
}

/** Where a range starts: the anchor while it is in `order`, else the current photo, else nowhere. */
function rangeStart(s: SelectionState, o: SelectionOrder): string | null {
  if (s.anchorId !== null && o.order.includes(s.anchorId)) return s.anchorId;
  return s.currentId !== null && o.order.includes(s.currentId) ? s.currentId : null;
}

/** `next`, or `s` itself when they are equal, so the store can skip a re-render. */
function settle(s: SelectionState, next: SelectionState): SelectionState {
  const same = next.currentId === s.currentId && next.anchorId === s.anchorId
    && next.selectedIds.length === s.selectedIds.length && next.selectedIds.every((id, i) => id === s.selectedIds[i]);
  return same ? s : next;
}

/** Shift wins over meta when both are held. A shift-click with nowhere to start from acts as a plain click. */
export function clickSelect(s: SelectionState, o: SelectionOrder, id: string, mods: { meta: boolean; shift: boolean }): SelectionState {
  if (mods.shift) {
    const start = rangeStart(s, o);
    if (start !== null && o.order.includes(id)) return settle(s, { currentId: id, selectedIds: range(o, start, id), anchorId: start });
  } else if (mods.meta) {
    const selectedIds = s.selectedIds.includes(id) ? s.selectedIds.filter((x) => x !== id) : [...s.selectedIds, id];
    return { ...s, selectedIds, anchorId: id };
  }
  return settle(s, { currentId: id, selectedIds: [id], anchorId: id });
}

/** Step `delta` shown photos from the current one, clamped to the ends; `extend` grows the range from the anchor. */
export function moveCurrent(s: SelectionState, o: SelectionOrder, delta: number, extend: boolean): SelectionState {
  const current = s.currentId;
  const at = current === null ? -1 : o.order.indexOf(current);
  if (current === null || at < 0 || delta === 0) return s;
  const list = shownIds(o);
  const before = o.order.slice(0, at).filter((id) => o.shown.has(id)).length;
  let i = before + delta;
  if (!o.shown.has(current)) {
    // A hidden current: the first shown photo in the move direction is the first step.
    if (delta > 0 ? before === list.length : before === 0) return s;
    if (delta > 0) i -= 1;
  }
  const target = list[Math.min(Math.max(i, 0), list.length - 1)];
  if (target === current) return s;
  if (!extend) return { currentId: target, selectedIds: [target], anchorId: target };
  const start = rangeStart(s, o) ?? current;
  return { currentId: target, selectedIds: range(o, start, target), anchorId: start };
}

export function selectAllShown(s: SelectionState, o: SelectionOrder): SelectionState {
  return settle(s, { ...s, selectedIds: shownIds(o) });
}

/** Clears when every shown photo is already selected, else selects all shown. */
export function toggleSelectAll(s: SelectionState, o: SelectionOrder): SelectionState {
  const selected = new Set(s.selectedIds);
  return shownIds(o).every((id) => selected.has(id)) ? clearSelection(s) : selectAllShown(s, o);
}

export function clearSelection(s: SelectionState): SelectionState {
  return s.selectedIds.length === 0 ? s : { ...s, selectedIds: [] };
}

/** Drops ids that are no longer present from the selection and the anchor; the current photo is left alone. */
export function prune(s: SelectionState, present: ReadonlySet<string>): SelectionState {
  const anchorId = s.anchorId !== null && present.has(s.anchorId) ? s.anchorId : null;
  return settle(s, { ...s, selectedIds: s.selectedIds.filter((id) => present.has(id)), anchorId });
}
