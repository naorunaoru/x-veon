import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from '@/components/ui/dialog';
import { useAppStore } from '@/store';
import { requestWindowManagementHeadroom } from '@/gl/hdr-display';

export function HdrPermissionDialog() {
  const needed = useAppStore((s) => s.hdrPermissionNeeded);
  const setNeeded = useAppStore((s) => s.setHdrPermissionNeeded);
  const setDisplayHdr = useAppStore((s) => s.setDisplayHdr);
  const [requesting, setRequesting] = useState(false);

  async function handleAllow() {
    setRequesting(true);
    const headroom = await requestWindowManagementHeadroom();
    setRequesting(false);
    if (headroom != null) {
      setDisplayHdr(true, headroom);
    }
    setNeeded(false);
  }

  return (
    <Dialog open={needed} onOpenChange={(v) => { if (!v) setNeeded(false); }}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>HDR Display Detection</DialogTitle>
        </DialogHeader>

        <div className="space-y-2 py-2 text-sm text-muted-foreground">
          <p>
            Your display appears to support HDR. To determine its exact peak
            brightness, the app needs permission to query display information
            via the Window Management API.
          </p>
          <p>
            Without this, a conservative default will be used which may not
            fully utilize your display's HDR capability.
          </p>
        </div>

        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => setNeeded(false)}>
            Skip
          </Button>
          <Button size="sm" onClick={handleAllow} disabled={requesting}>
            Allow
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
