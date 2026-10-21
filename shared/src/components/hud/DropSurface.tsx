import { RAW_ACCEPT } from '@/lib/catalog';
import { importFiles } from '@/app/services/library';
import { useCallback, useRef } from 'react';
import { Upload } from 'lucide-react';
import './DropSurface.css';

// Formats we test and stand behind. The picker also accepts others (see RAW_EXTENSIONS) so an
// unsupported file gets a clear decode error instead of being ignored; DNG and CR3 (no decoder
// in the vendored rawloader) aren't advertised.
const FORMAT_LINE = 'RAF · ARW · NEF · CR2';

type DropSurfaceProps = {
  /** false = solid empty canvas; true = scrim+blur overlay over the live view. */
  overlay?: boolean;
  /** Lights the frame primary + corner ticks + glyph bob. Always true for the
   *  overlay; only while dragging for the empty state. */
  active?: boolean;
  /** Overlay only — count shown in the "appending to N photos" pill. */
  fileCount?: number;
  /** Empty state only — models and WASM are still loading: a spinner instead of
   *  the drop prompt, and no file picker. */
  loading?: boolean;
};

const CORNERS = ['tl', 'tr', 'bl', 'br'] as const;

export function DropSurface({ overlay = false, active = false, fileCount = 0, loading = false }: DropSurfaceProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const onPick = useCallback((list: FileList | null) => {
    if (list) importFiles(Array.from(list));
  }, [importFiles]);

  const pickable = !overlay && !loading;
  const className = [
    'xv-drop',
    overlay ? 'xv-drop--overlay' : 'xv-drop--empty',
    loading ? 'xv-drop--loading' : '',
    active ? 'is-active' : '',
  ].filter(Boolean).join(' ');

  return (
    <div
      data-testid={overlay ? 'drop-overlay' : 'empty-dropzone'}
      className={className}
      onClick={pickable ? () => inputRef.current?.click() : undefined}
    >
      <div className="xv-drop__frame" />
      {active && CORNERS.map((c) => <div key={c} className={`xv-drop__tick xv-drop__tick--${c}`} />)}

      {loading ? (
        <div className="xv-drop__center">
          <div className="xv-drop__spinner" />
          <div className="xv-drop__loading">Loading models and WASM…</div>
        </div>
      ) : (
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
      )}

      {pickable && (
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
