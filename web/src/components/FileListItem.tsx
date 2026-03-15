import { Loader2, ImageIcon, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { QueuedFile } from '@/store';

interface FileListItemProps {
  file: QueuedFile;
  selected: boolean;
  onSelect: () => void;
  onRemove: () => void;
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
      <div className="h-14 w-14 rounded bg-muted flex-shrink-0 overflow-hidden relative">
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
        {file.status === 'processing' && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/50">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
          </div>
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{file.originalName}</p>
        <p className="text-xs text-muted-foreground truncate">
          {file.metadata?.camera ?? '\u2014'}
        </p>
        {file.metadata?.lensModel && (
          <p className="text-xs text-muted-foreground truncate">
            {file.metadata.lensModel}
            {file.metadata.focalLength > 0 && ` ${Math.round(file.metadata.focalLength)}mm`}
            {file.metadata.fNumber > 0 && ` \u0192/${parseFloat(file.metadata.fNumber.toFixed(1))}`}
          </p>
        )}
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
    </div>
  );
}
