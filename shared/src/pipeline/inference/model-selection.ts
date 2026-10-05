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
  // Only `{cfa}_w{width}_base` entries are models the app may pick.
  const keyOf = (size: ModelSize) => `${cfa}_w${widths[size]}_base`;
  const sizeOf = (key: string): ModelSize | undefined =>
    (Object.keys(widths) as ModelSize[]).find((size) => key === keyOf(size));
  const exact =
    recorded &&
    Object.keys(manifest).find((key) => sizeOf(key) && manifest[key].source_sha256 === recorded.sha256);
  const bySize = (size: ModelSize) => (manifest[keyOf(size)] ? keyOf(size) : undefined);
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
