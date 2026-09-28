import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import type { Host } from '@/host';
import { setHost } from './app/services/host';
import App from './App';
import './index.css';
import './styles/tokens.css';
import './styles/glass.css';
export function startApp(root: HTMLElement, host: Host): () => void {
  setHost(host);
  const reactRoot = createRoot(root);
  reactRoot.render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
  return () => reactRoot.unmount();
}
