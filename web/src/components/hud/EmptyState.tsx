import { useCallback, useRef, useState } from 'react';
import { Upload } from 'lucide-react';
import { useAppStore } from '@/store';
import './EmptyState.css';

const RAW_ACCEPT = '.raf,.cr2,.cr3,.nef,.nrw,.arw,.dng,.rw2,.orf,.pef,.srw,.erf,.kdc,.dcr,.mef';

export function EmptyState() {
  const addFiles = useAppStore((s) => s.addFiles);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const onPick = useCallback((list: FileList | null) => {
    if (list) addFiles(Array.from(list));
  }, [addFiles]);

  return (
    <div
      data-testid="empty-dropzone"
      className={`xv-empty${dragging ? ' is-dragging' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false); }}
      onDrop={(e) => { e.preventDefault(); setDragging(false); onPick(e.dataTransfer.files); }}
    >
      <div className="xv-empty__glyph"><Upload size={28} /></div>
      <div className="xv-empty__title">Drop RAW files here</div>
      <div className="xv-empty__sub">RAF · NEF · ARW · CR3 · DNG · everything local</div>
      <div className="xv-empty__browse">or <b>browse files</b></div>
      <input ref={inputRef} type="file" accept={RAW_ACCEPT} multiple hidden onChange={(e) => onPick(e.target.files)} />
    </div>
  );
}
