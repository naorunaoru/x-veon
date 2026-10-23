import { useCallback, useEffect, useRef, useState } from 'react';

/** Open state, outside-click and Escape close, and focus return for a trigger + role="dialog" popover. */
export function usePopover() {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const close = useCallback(() => { setOpen(false); triggerRef.current?.focus(); }, []);
  useEffect(() => {
    if (!open) return;
    popoverRef.current?.focus();
    const onPointerDown = (event: PointerEvent) => {
      const inside = [triggerRef, popoverRef].some((ref) => ref.current?.contains(event.target as Node));
      if (!inside) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      close();
    };
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open, close]);
  return {
    open, close,
    triggerProps: { ref: triggerRef, 'aria-haspopup': 'dialog' as const, 'aria-expanded': open, onClick: () => setOpen((value) => !value) },
    popoverProps: { ref: popoverRef, role: 'dialog', tabIndex: -1 },
  };
}
