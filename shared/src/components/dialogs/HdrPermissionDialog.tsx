import { useState } from 'react';
import { getHost } from '@/app/services/host';
import { useAppStore } from '@/app/store';
import { Dialog, DialogContent } from './Dialog';

export function HdrPermissionDialog() {
  const needed = useAppStore((state) => state.hdrPermissionNeeded);
  const setNeeded = useAppStore((state) => state.setHdrPermissionNeeded);
  const setDisplayHdr = useAppStore((state) => state.setDisplayHdr);
  const [requesting, setRequesting] = useState(false);

  async function handleAllow() {
    setRequesting(true);
    const headroom = await getHost().display.requestAccurateHeadroom?.();
    setRequesting(false);
    if (headroom != null) setDisplayHdr(true, headroom);
    setNeeded(false);
  }

  return (
    <Dialog open={needed && !!getHost().display.requestAccurateHeadroom} onOpenChange={(value) => { if (!value) setNeeded(false); }}>
      <DialogContent
        title="HDR Display Detection"
        actions={(
          <>
            <button type="button" className="xv-btn" onClick={() => setNeeded(false)}>
              Skip
            </button>
            <button
              type="button" className="xv-btn xv-btn--primary"
              onClick={handleAllow} disabled={requesting}
            >
              Allow
            </button>
          </>
        )}
      >
        <p>
          Your display appears to support HDR. To determine its exact peak brightness, the app
          needs permission to query display information via the Window Management API.
        </p>
        <p>
          Without this, a conservative default will be used which may not fully utilize your
          display&apos;s HDR capability.
        </p>
      </DialogContent>
    </Dialog>
  );
}
