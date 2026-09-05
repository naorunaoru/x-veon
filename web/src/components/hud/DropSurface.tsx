import { RAW_ACCEPT } from '@/lib/catalog';
import { importFiles } from '@/app/services/library';
import { useCallback, useRef } from 'react';
import { Upload } from 'lucide-react';
import './DropSurface.css';

const FORMAT_LINE = 'RAF · NEF · ARW · CR3 · DNG';

type DropSurfaceProps = {
  /** false = solid empty canvas; true = scrim+blur overlay over the live view. */
  overlay?: boolean;
  /** Lights the frame primary + corner ticks + glyph bob. Always true for the
   *  overlay; only while dragging for the empty state. */
  active?: boolean;
  /** Overlay only — count shown in the "appending to N photos" pill. */
  fileCount?: number;
};

const CORNERS = ['tl', 'tr', 'bl', 'br'] as const;

export function DropSurface({ overlay = false, active = false, fileCount = 0 }: DropSurfaceProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const onPick = useCallback((list: FileList | null) => {
    if (list) importFiles(Array.from(list));
  }, [importFiles]);

  const className = [
    'xv-drop',
    overlay ? 'xv-drop--overlay' : 'xv-drop--empty',
    active ? 'is-active' : '',
  ].filter(Boolean).join(' ');

  return (
    <div
      data-testid={overlay ? 'drop-overlay' : 'empty-dropzone'}
      className={className}
      onClick={overlay ? undefined : () => inputRef.current?.click()}
    >
      <div className="xv-drop__frame" />
      {active && CORNERS.map((c) => <div key={c} className={`xv-drop__tick xv-drop__tick--${c}`} />)}

      <div className="xv-drop__center">
        <div className="xv-drop__glyph"><Upload size={30} strokeWidth={1.5} /></div>
        <div className="xv-drop__title">
          {overlay ? 'Drop to add to this session' : 'Drop RAW files to start'}
        </div>
        <div className="xv-drop__format">{FORMAT_LINE}</div>

        {overlay ? (
          <div className="xv-drop__pill">
            <span className="xv-drop__dot" />
            appending to {fileCount} photo{fileCount === 1 ? '' : 's'} already loaded
          </div>
        ) : (
          <div className="xv-drop__browse">
            or <span className="xv-drop__browse-link">browse files</span>
          </div>
        )}
      </div>

      {!overlay && (
        <input
          ref={inputRef}
          type="file"
          accept={RAW_ACCEPT}
          multiple
          hidden
          onChange={(e) => onPick(e.target.files)}
        />
      )}
    </div>
  );
}
