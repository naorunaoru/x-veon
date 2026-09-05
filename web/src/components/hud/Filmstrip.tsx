import { importFiles, removeFile } from '@/app/services/library';
import { useCallback, useRef } from 'react';
import { Plus, X } from 'lucide-react';
import { useAppStore } from '@/app/store';
import './Filmstrip.css';

const RAW_ACCEPT = '.raf,.cr2,.cr3,.nef,.nrw,.arw,.dng,.rw2,.orf,.pef,.srw,.erf,.kdc,.dcr,.mef';

export function Filmstrip() {
  const files = useAppStore((s) => s.files);
  const selectedFileId = useAppStore((s) => s.selectedFileId);
  const selectFile = useAppStore((s) => s.selectFile);
  const inputRef = useRef<HTMLInputElement>(null);

  const onPick = useCallback((list: FileList | null) => {
    if (list) importFiles(Array.from(list));
  }, [importFiles]);

  return (
    <div className="xv-filmstrip xv-glass">
      <span className="xv-filmstrip__count">{files.length} {files.length === 1 ? 'file' : 'files'}</span>
      <div className="xv-filmstrip__scroll">
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
            <button
              className="xv-thumb__remove"
              aria-label="Remove file"
              onClick={(e) => { e.stopPropagation(); removeFile(f.id); }}
            >
              <X size={9} />
            </button>
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
