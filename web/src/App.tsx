import { Sidebar } from './components/Sidebar';
import { OutputPanel } from './components/OutputPanel';
import { GradingPanel } from './components/GradingPanel';
import { useInit } from './hooks/useInit';
import { useAppStore } from './store';

export default function App() {
  useInit();

  const selectedFileId = useAppStore((s) => s.selectedFileId);
  const hasResult = useAppStore((s) => {
    const f = s.files.find((f) => f.id === s.selectedFileId);
    return f?.status === 'done';
  });

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Sidebar />
      {selectedFileId && <OutputPanel />}
      {hasResult && <GradingPanel />}
    </div>
  );
}
