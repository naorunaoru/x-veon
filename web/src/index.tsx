import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from '@/App';
import '@/index.css';
import '@/styles/tokens.css';
import '@/styles/glass.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

if (__XV_GOLDEN__ && new URLSearchParams(location.search).has('golden')) {
  import('@/dev/golden')
    .then((module) => module.runGolden())
    .catch((error) => console.error('[golden] failed:', error));
}
