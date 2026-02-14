import { useCallback, useRef, useState } from 'react';
import { Upload } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';

interface DropZoneProps {
  className?: string;
}

export function DropZone({ className }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const addFiles = useAppStore((s) => s.addFiles);

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

  const onDragLeave = useCallback(() => {
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
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={cn(
        'flex flex-col items-center justify-center gap-3',
        'border-2 border-dashed border-muted-foreground/25 rounded-lg m-4',
        'text-muted-foreground cursor-pointer transition-colors',
        isDragging && 'border-primary bg-primary/5',
        className,
      )}
    >
      <Upload className="h-12 w-12" />
      <p className="text-lg">Drop RAW files here</p>
      <p className="text-sm underline">or click to browse</p>
      <input
        ref={inputRef}
        type="file"
        accept=".raf,.cr2,.cr3,.nef,.nrw,.arw,.dng,.rw2,.orf,.pef,.srw,.erf,.kdc,.dcr,.mef"
        multiple
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
