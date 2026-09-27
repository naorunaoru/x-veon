import { HudRoot } from './components/hud/HudRoot';
import { HdrPermissionDialog } from './components/dialogs/HdrPermissionDialog';
import { useBootstrap } from './app/hooks/useBootstrap';
import { useAutoProcess } from './app/hooks/useAutoProcess';

export default function App() {
  useBootstrap();
  useAutoProcess();

  return (
    <>
      <HudRoot />
      <HdrPermissionDialog />
    </>
  );
}
