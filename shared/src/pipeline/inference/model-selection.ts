import type { CfaType, ModelIdentity, ModelSize } from '@/lib/types';
interface Meta {
  source_sha256?: string;
  base_width?: number;
}
const widths: Record<ModelSize, number> = { S: 16, M: 32, L: 64 };
export function resolveModel(
  manifest: Record<string, Meta>,
  cfa: CfaType,
  recorded: ModelIdentity | null,
  defaultSize: ModelSize,
): { key: string; model: ModelIdentity; note: string | null } {
  const sizeOf = (key: string): ModelSize | undefined =>
    (Object.keys(widths) as ModelSize[]).find(
      (size) => manifest[key].base_width === widths[size] || key.startsWith(`${cfa}_w${widths[size]}_`),
    );
  const exact =
    recorded &&
    Object.keys(manifest).find(
      (key) => key.startsWith(`${cfa}_`) && manifest[key].source_sha256 === recorded.sha256 && sizeOf(key),
    );
  const bySize = (size: ModelSize) =>
    [`${cfa}_w${widths[size]}_hl`, `${cfa}_w${widths[size]}_base`].find((key) => manifest[key]);
  const key = exact || (recorded && bySize(recorded.size)) || bySize(defaultSize);
  if (!key) throw new Error(`No ${recorded?.size ?? defaultSize} model is available for ${cfa}`);
  const sha256 = manifest[key].source_sha256;
  if (!sha256) throw new Error(`Model ${key} has no checkpoint hash.`);
  return {
    key,
    model: { size: sizeOf(key)!, sha256 },
    note:
      recorded && !exact
        ? 'This edit was made with a different model. This build’s available model is shown.'
        : null,
  };
}
