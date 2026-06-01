import './Toggle.css';

interface ToggleProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
}

export function Toggle({ checked, onChange, label }: ToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className={`xv-switch${checked ? ' is-on' : ''}`}
      onClick={() => onChange(!checked)}
    >
      <span className="xv-switch__knob" />
    </button>
  );
}
