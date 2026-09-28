import { setHost } from '@/app/services/host';
import { fakeHost } from '@/test/fake-host';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { HdrPermissionDialog } from './HdrPermissionDialog';
import { useAppStore } from '@/app/store';

describe('HdrPermissionDialog', () => {
  beforeEach(() => {
    const host = fakeHost();
    host.display.requestAccurateHeadroom = vi.fn(async () => 2.5);
    setHost(host);
    useAppStore.setState({
      hdrPermissionNeeded: true,
      displayHdr: false,
      displayHdrHeadroom: 1,
    });
  });

  it('clears the permission flag when skipped', () => {
    render(<HdrPermissionDialog />);
    fireEvent.click(screen.getByRole('button', { name: 'Skip' }));
    expect(useAppStore.getState().hdrPermissionNeeded).toBe(false);
  });

  it('requests the headroom and enables HDR when allowed', async () => {
    render(<HdrPermissionDialog />);
    fireEvent.click(screen.getByRole('button', { name: 'Allow' }));
    await waitFor(() => expect(useAppStore.getState().displayHdr).toBe(true));
    expect(useAppStore.getState().displayHdrHeadroom).toBe(2.5);
    expect(useAppStore.getState().hdrPermissionNeeded).toBe(false);
  });
});
