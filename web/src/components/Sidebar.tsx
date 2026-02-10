import { Header } from './Header';
import { FileList } from './FileList';
import { DropZone } from './DropZone';
import { SettingsPanel } from './SettingsPanel';
import { useAppStore } from '@/store';

export function Sidebar() {
  const hasFiles = useAppStore((s) => s.files.length > 0);

  return (
    <aside className="flex flex-col h-screen w-[480px] min-w-[480px] border-r border-border">
      <Header />
      <div className="flex-1 overflow-hidden flex flex-col">
        {hasFiles ? <FileList /> : <DropZone className="flex-1" />}
      </div>
      <SettingsPanel />
    </aside>
  );
}
