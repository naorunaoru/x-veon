import { useEffect, useState } from 'react';
import { getHost } from '@/app/services/host';
import './UpdateNotice.css';

export function UpdateNotice() {
  const [release, setRelease] = useState<{ name: string; version: string; url: string } | null>(null);
  useEffect(() => {
    let live = true;
    void getHost().updates?.check().then(found => { if (live) setRelease(found); }, () => {});
    return () => { live = false; };
  }, []);
  if (!release) return null;
  return (
    <div className="xv-update-notice xv-glass" role="status">
      <p>{release.name} {release.version} is available.</p>
      <a className="xv-btn" href={release.url} target="_blank" rel="noreferrer">Download</a>
      <button type="button" className="xv-btn" onClick={() => setRelease(null)}>Dismiss</button>
    </div>
  );
}
