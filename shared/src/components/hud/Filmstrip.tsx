import { getHost } from '@/app/services/host';
import { RAW_ACCEPT } from '@/lib/catalog';
import { importFiles, removeFile } from '@/app/services/library';
import { useCallback, useEffect, useRef } from 'react';
import { Plus, X } from 'lucide-react';
import { useAppStore } from '@/app/store';
import './Filmstrip.css';


export function Filmstrip() {
  const files = useAppStore((s) => s.files);
  const selectedFileId = useAppStore((s) => s.selectedFileId);
  const selectFile = useAppStore((s) => s.selectFile);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const strip = scrollRef.current;
    if (!strip) return;
    const onWheel = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey || Math.abs(event.deltaX) >= Math.abs(event.deltaY)) return;
      if (strip.scrollWidth <= strip.clientWidth) return;
      event.preventDefault();
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? strip.clientWidth : 1;
      strip.scrollLeft += event.deltaY * unit;
    };
    strip.addEventListener('wheel', onWheel, { passive: false });
    return () => strip.removeEventListener('wheel', onWheel);
  }, []);

  const onPick = useCallback((list: FileList | null) => {
    if (list) importFiles(Array.from(list));
  }, [importFiles]);

  return (
    <div className="xv-filmstrip xv-glass">
      <span className="xv-filmstrip__count">{files.length} {files.length === 1 ? 'file' : 'files'}</span>
      <div className="xv-filmstrip__scroll" ref={scrollRef}>
        {files.map((f, i) => (
          <div
            key={f.id}
            data-testid="filmstrip-thumb"
            className={`xv-thumb${f.id === selectedFileId ? ' is-selected' : ''}`}
            onClick={() => selectFile(f.id)}
          >
            {f.thumbnailUrl && <img src={f.thumbnailUrl} alt={f.originalName} />}
            <span className={`xv-thumb__dot ${f.status}`} />
            <span className="xv-thumb__index">{String(i + 1).padStart(2, '0')}</span>
            {getHost().library.remove && <button
              className="xv-thumb__remove"
              aria-label="Remove file"
              onClick={(e) => { e.stopPropagation(); removeFile(f.id); }}
            >
              <X size={9} />
            </button>}
          </div>
        ))}
        <button className="xv-filmstrip__add" aria-label="Add files" onClick={() => inputRef.current?.click()}>
          <Plus size={14} />
        </button>
      </div>
      <input
        ref={inputRef} type="file" accept={RAW_ACCEPT} multiple hidden
        onChange={(e) => onPick(e.target.files)}
      />
    </div>
  );
}
