import { ChevronDown } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useAppStore } from '@/app/store';
import { FILTER_LABELS, RATING_FILTERS, filterCounts } from '@/app/rating';
import { useMenu } from './useMenu';
// The menu reuses the folder menu's list and item styles.
import './TopBar.css';

/** The column's top row: the rating filter, with counts over every photo in the folder. */
export function FilterMenu() {
  const filter = useAppStore((s) => s.filter);
  const setFilter = useAppStore((s) => s.setFilter);
  const ratings = useAppStore(useShallow((s) => s.files.map((f) => f.rating)));
  const counts = filterCounts(ratings);
  const menu = useMenu();

  return (
    <div className="xv-column__top" ref={menu.rootRef}>
      <button type="button" className="xv-column__filter" aria-label="Filter photos" {...menu.triggerProps}>
        {`${FILTER_LABELS[filter]} · ${counts[filter]}`}
        <ChevronDown size={10} />
      </button>
      {menu.open && (
        <div className="xv-folder-menu__list xv-glass-heavy" {...menu.menuProps}>
          {RATING_FILTERS.map((f) => (
            <button
              key={f}
              type="button"
              role="menuitemradio"
              aria-checked={f === filter}
              className="xv-folder-menu__item"
              onClick={() => { setFilter(f); menu.close(); }}
            >
              {`${FILTER_LABELS[f]} · ${counts[f]}`}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
