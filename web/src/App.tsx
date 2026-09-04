import { HudRoot } from './components/hud/HudRoot';
import { HdrPermissionDialog } from './components/dialogs/HdrPermissionDialog';
import { useInit } from './hooks/useInit';
import { useAutoProcess } from './hooks/useAutoProcess';

export default function App() {
  useInit();
  useAutoProcess();

  return (
    <>
      <HudRoot />
      <HdrPermissionDialog />
    </>
  );
}
