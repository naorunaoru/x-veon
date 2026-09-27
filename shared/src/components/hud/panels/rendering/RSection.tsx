import type { ReactNode } from 'react';
import { RotateCcw } from 'lucide-react';

interface RSectionProps {
  title: string;
  modified?: boolean;
  onReset?: () => void;
  children: ReactNode;
}

/** A titled section block in the Rendering panel body, with a modified dot
 *  and a reset affordance that appears only when the section is modified. */
export function RSection({ title, modified = false, onReset, children }: RSectionProps) {
  return (
    <section className="xv-rsection">
      <div className="xv-rsection__head">
        <span className="xv-rsection__label">{title}</span>
        {modified && <span className="xv-rsection__dot" />}
        {modified && onReset && (
          <button type="button" className="xv-rsection__reset" aria-label={`Reset ${title}`} onClick={onReset}>
            <RotateCcw size={11} />
          </button>
        )}
      </div>
      {children}
    </section>
  );
}
