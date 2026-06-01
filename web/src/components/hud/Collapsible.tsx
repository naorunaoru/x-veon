import { useState, type ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';
import './Collapsible.css';

interface CollapsibleProps {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

export function Collapsible({ title, defaultOpen = false, children }: CollapsibleProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="xv-collapsible">
      <button type="button" className="xv-collapsible__head" onClick={() => setOpen((o) => !o)}>
        <ChevronRight size={12} className={`xv-collapsible__chevron${open ? ' is-open' : ''}`} />
        <span className="xv-collapsible__title">{title}</span>
      </button>
      {open && <div className="xv-collapsible__body">{children}</div>}
    </section>
  );
}
