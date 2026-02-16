import { Loader2 } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { OutputCanvas } from './OutputCanvas';
import { useAppStore } from '@/store';

export function OutputPanel() {
  const selectedFile = useAppStore((s) =>
    s.files.find((f) => f.id === s.selectedFileId),
  );
  if (!selectedFile) {
    return (
      <main className="flex-1 flex items-center justify-center bg-background">
        <p className="text-muted-foreground">Select a file from the list</p>
      </main>
    );
  }

  const isReprocessing = selectedFile.status === 'processing' && !!selectedFile.result;

  return (
    <main className="flex-1 flex bg-background overflow-hidden relative">
      {selectedFile.result ? (
        <>
          <OutputCanvas key={selectedFile.id} fileId={selectedFile.id} result={selectedFile.result} />
          {isReprocessing && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/50 pointer-events-none">
              <div className="flex flex-col items-center gap-4 w-64">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                {selectedFile.progress && (
                  <Progress
                    value={(selectedFile.progress.current / selectedFile.progress.total) * 100}
                    label={`${selectedFile.progress.current}/${selectedFile.progress.total} tiles`}
                    className="w-full"
                  />
                )}
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          {selectedFile.status === 'processing' ? (
            <div className="flex flex-col items-center gap-4 w-64">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              {selectedFile.progress && (
                <Progress
                  value={(selectedFile.progress.current / selectedFile.progress.total) * 100}
                  label={`${selectedFile.progress.current}/${selectedFile.progress.total} tiles`}
                  className="w-full"
                />
              )}
            </div>
          ) : selectedFile.status === 'error' ? (
            <p className="text-destructive">{selectedFile.error}</p>
          ) : (
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          )}
        </div>
      )}
    </main>
  );
}
