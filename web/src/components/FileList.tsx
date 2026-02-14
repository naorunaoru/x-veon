import { useCallback, useRef, useState } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { FileListItem } from './FileListItem';
import { useAppStore } from '@/store';
import { cn } from '@/lib/utils';

export function FileList() {
  const files = useAppStore((s) => s.files);
  const selectedFileId = useAppStore((s) => s.selectedFileId);
  const selectFile = useAppStore((s) => s.selectFile);
  const addFiles = useAppStore((s) => s.addFiles);
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = useCallback(
    (fileList: FileList | null) => {
      if (!fileList) return;
      addFiles(Array.from(fileList));
    },
    [addFiles],
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    // Only clear when leaving the container, not when entering a child
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragging(false);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles],
  );

  return (
    <div
      className="flex-1 overflow-hidden relative"
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <ScrollArea className="h-full">
        <div className="p-2 space-y-1">
          {files.map((f) => (
            <FileListItem
              key={f.id}
              file={f}
              selected={f.id === selectedFileId}
              onSelect={() => selectFile(f.id)}
            />
          ))}
          <button
            onClick={() => inputRef.current?.click()}
            className="w-full p-2 text-sm text-muted-foreground hover:text-foreground transition-colors text-center"
          >
            + Add more files
          </button>
          <input
            ref={inputRef}
            type="file"
            accept=".raf,.cr2,.cr3,.nef,.nrw,.arw,.dng,.rw2,.orf,.pef,.srw,.erf,.kdc,.dcr,.mef"
            multiple
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      </ScrollArea>
      {/* Drag overlay */}
      {isDragging && (
        <div
          className={cn(
            'absolute inset-0 flex items-center justify-center',
            'bg-primary/5 border-2 border-dashed border-primary rounded-lg',
            'text-primary text-sm font-medium pointer-events-none',
          )}
        >
          Drop RAW files here
        </div>
      )}
    </div>
  );
}
