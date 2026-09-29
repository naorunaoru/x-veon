import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
const root = process.cwd(),
  evidence = path.join(root, 'tmp/m2-spike/browser');
await mkdir(evidence, { recursive: true });
const capture = `<script>const t=setInterval(()=>{const p=window.__spike||window.__golden;if(!p)return;clearInterval(t);fetch('/report/'+(new URLSearchParams(location.search).get('report')||'report'),{method:'POST',body:JSON.stringify(p)});},500);</script>`;
createServer(async (req, res) => {
  try {
    const u = new URL(req.url, 'http://localhost');
    if (req.method === 'POST' && /^\/report\/[a-z0-9-]+$/.test(u.pathname)) {
      let b = '';
      for await (const c of req) {
        b += c;
        if (b.length > 2_000_000) throw Error('Report too large');
      }
      JSON.parse(b);
      await writeFile(path.join(evidence, u.pathname.slice(8) + '.json'), b);
      res.end('saved');
      return;
    }
    const rel = decodeURIComponent(u.pathname.slice(1)) || 'index.html';
    if (rel.includes('\\') || rel.split('/').includes('..'))
      throw Error('Bad path');
    const file = path.join(root, 'web/dist', rel);
    const data = await readFile(file);
    const ext = path.extname(file);
    res.setHeader(
      'content-type',
      {
        '.html': 'text/html',
        '.js': 'text/javascript',
        '.css': 'text/css',
        '.wasm': 'application/wasm',
        '.json': 'application/json',
      }[ext] || 'application/octet-stream',
    );
    res.end(ext === '.html' ? data.toString() + capture : data);
  } catch (e) {
    res.writeHead(404).end(String(e));
  }
}).listen(39217, '127.0.0.1', async () => {
  await writeFile(path.join(evidence, 'server.pid'), String(process.pid));
  console.log('Task web server PID', process.pid);
});
