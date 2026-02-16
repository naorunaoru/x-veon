import { Circle, Loader2, CheckCircle2, AlertCircle, ImageIcon, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { QueuedFile } from '@/store';

interface FileListItemProps {
  file: QueuedFile;
  selected: boolean;
  onSelect: () => void;
  onRemove: () => void;
}

function StatusIcon({ status }: { status: QueuedFile['status'] }) {
  switch (status) {
    case 'queued':
      return <Circle className="h-4 w-4 text-muted-foreground" />;
    case 'processing':
      return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
    case 'done':
      return <CheckCircle2 className="h-4 w-4 text-green-500" />;
    case 'error':
      return <AlertCircle className="h-4 w-4 text-destructive" />;
  }
}

export function FileListItem({ file, selected, onSelect, onRemove }: FileListItemProps) {
  return (
    <div
      onClick={onSelect}
      className={cn(
        'flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-colors overflow-hidden',
        'hover:bg-accent',
        selected && 'bg-accent ring-1 ring-ring',
      )}
    >
      {/* Thumbnail */}
      <div className="h-14 w-14 rounded bg-muted flex-shrink-0 overflow-hidden">
        {file.thumbnailUrl ? (
          <img
            src={file.thumbnailUrl}
            alt={file.name}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="h-full w-full flex items-center justify-center text-muted-foreground">
            <ImageIcon className="h-6 w-6" />
          </div>
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{file.file.name}</p>
        <p className="text-xs text-muted-foreground truncate">
          {file.metadata?.camera ?? '\u2014'}
        </p>
        {file.error && (
          <p className="text-xs text-destructive truncate">{file.error}</p>
        )}
      </div>

      {/* Remove */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        className="flex-shrink-0 text-muted-foreground/50 hover:text-destructive transition-colors"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>

      {/* Status */}
      <div className="flex-shrink-0">
        <StatusIcon status={file.status} />
      </div>
    </div>
  );
}
