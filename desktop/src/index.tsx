import { startApp } from '@/startApp';
import { createDesktopHost } from './host';
import { runSpike } from './spike/run';
const fixture = createDesktopHost();
startApp(document.getElementById('root')!, fixture.host);
void runSpike(fixture.releaseFixture);
