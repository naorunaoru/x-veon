import { startApp } from '@/startApp';
import { createDesktopHost } from './host';
const root = document.getElementById('root')!;
if (__XV_GOLDEN__ && new URLSearchParams(location.search).has('golden')) {
  void import('./golden-runner').then(({ runGoldenApp }) => runGoldenApp(root));
} else startApp(root, createDesktopHost(window.xveon));
