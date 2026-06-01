import { useState, type ReactNode } from 'react';
import { flushSync } from 'react-dom';
import { ChevronRight } from 'lucide-react';
import { Toggle } from './Toggle';
import './Collapsible.css';

interface CollapsibleProps {
  title: string;
  defaultOpen?: boolean;
  /** When provided, renders an enable switch in the header (controlled). */
  enabled?: boolean;
  onToggle?: (next: boolean) => void;
  children: ReactNode;
}

export function Collapsible({ title, defaultOpen = false, enabled, onToggle, children }: CollapsibleProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="xv-collapsible">
      <button type="button" className="xv-collapsible__head" onClick={() => flushSync(() => setOpen((o) => !o))}>
        <ChevronRight size={12} className={`xv-collapsible__chevron${open ? ' is-open' : ''}`} />
        <span className="xv-collapsible__title">{title}</span>
        {enabled !== undefined && onToggle && (
          <span onClick={(e) => e.stopPropagation()}>
            <Toggle checked={enabled} onChange={onToggle} label={title} />
          </span>
        )}
      </button>
      {open && <div className="xv-collapsible__body">{children}</div>}
    </section>
  );
}
