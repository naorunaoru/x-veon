import { HudRoot } from './components/hud/HudRoot';
import { HdrPermissionDialog } from './components/dialogs/HdrPermissionDialog';
import { useInit } from './app/hooks/useInit';
import { useAutoProcess } from './app/hooks/useAutoProcess';

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
