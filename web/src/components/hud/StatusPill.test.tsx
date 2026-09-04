import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusPill } from './StatusPill';
import { useAppStore } from '@/app/store';

function setStore(partial: Partial<ReturnType<typeof useAppStore.getState>>) {
  useAppStore.setState(partial);
}

describe('StatusPill', () => {
  beforeEach(() => {
    setStore({ initialized: false, initError: null, backend: null, displayHdr: false });
  });

  it('shows a loading message before init', () => {
    render(<StatusPill />);
    expect(screen.getByText(/Loading models and WASM/i)).toBeInTheDocument();
  });

  it('shows the backend name once initialized', () => {
    setStore({ initialized: true, backend: 'webgpu' });
    render(<StatusPill />);
    expect(screen.getByText('webgpu')).toBeInTheDocument();
  });

  it('appends HDR when display HDR is on', () => {
    setStore({ initialized: true, backend: 'webgpu', displayHdr: true });
    render(<StatusPill />);
    expect(screen.getByText('webgpu · HDR')).toBeInTheDocument();
  });

  it('shows a raw init error', () => {
    setStore({ initError: 'WebGPU adapter not available' });
    render(<StatusPill />);
    expect(screen.getByText(/Init failed: WebGPU adapter not available/)).toBeInTheDocument();
  });
});
