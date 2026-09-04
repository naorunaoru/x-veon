import type { ReactNode } from 'react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import './Dialog.css';

export const Dialog = DialogPrimitive.Root;

interface DialogContentProps {
  title: string;
  children: ReactNode;
  actions?: ReactNode;
}

/** Centred glass panel with a title, body, and optional footer actions. */
export function DialogContent({ title, children, actions }: DialogContentProps) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="xv-dialog__overlay" />
      <DialogPrimitive.Content
        className="xv-dialog xv-glass-heavy"
        aria-describedby={undefined}
      >
        <DialogPrimitive.Title className="xv-dialog__title">{title}</DialogPrimitive.Title>
        <div className="xv-dialog__body">{children}</div>
        {actions && <div className="xv-dialog__actions">{actions}</div>}
        <DialogPrimitive.Close className="xv-dialog__close" aria-label="Close">
          <X size={14} />
        </DialogPrimitive.Close>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}
