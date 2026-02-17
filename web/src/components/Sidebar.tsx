import { Loader2 } from 'lucide-react';
import { Header } from './Header';
import { FileList } from './FileList';
import { DropZone } from './DropZone';
import { SettingsPanel } from './SettingsPanel';
import { useAppStore } from '@/store';

export function Sidebar() {
  const initialized = useAppStore((s) => s.initialized);
  const hasFiles = useAppStore((s) => s.files.length > 0);

  return (
    <aside className="flex flex-col h-screen w-[480px] min-w-[480px] border-r border-border">
      <Header />
      <div className="flex-1 overflow-hidden flex flex-col">
        {!initialized && !hasFiles ? (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : hasFiles ? (
          <FileList />
        ) : (
          <DropZone className="flex-1" />
        )}
      </div>
      <SettingsPanel />
    </aside>
  );
}
