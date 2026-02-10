import { ScrollArea } from '@/components/ui/scroll-area';
import { FileListItem } from './FileListItem';
import { DropZone } from './DropZone';
import { useAppStore } from '@/store';

export function FileList() {
  const files = useAppStore((s) => s.files);
  const selectedFileId = useAppStore((s) => s.selectedFileId);
  const selectFile = useAppStore((s) => s.selectFile);

  return (
    <ScrollArea className="flex-1">
      <div className="p-2 space-y-1">
        {files.map((f) => (
          <FileListItem
            key={f.id}
            file={f}
            selected={f.id === selectedFileId}
            onSelect={() => selectFile(f.id)}
          />
        ))}
        <DropZone compact className="mx-2 my-2" />
      </div>
    </ScrollArea>
  );
}
