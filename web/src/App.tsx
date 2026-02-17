import { Sidebar } from './components/Sidebar';
import { OutputPanel } from './components/OutputPanel';
import { GradingPanel } from './components/GradingPanel';
import { useInit } from './hooks/useInit';
import { useAppStore } from './store';

export default function App() {
  useInit();

  const selectedFileId = useAppStore((s) => s.selectedFileId);

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Sidebar />
      {selectedFileId && <OutputPanel />}
      {selectedFileId && <GradingPanel />}
    </div>
  );
}
