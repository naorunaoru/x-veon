/**
 * Convert LensFun XML database to JSON files for the web app.
 *
 * Clones the LensFun repo (shallow) if not already present, parses all
 * XML database files, and writes:
 *   public/lensfun/index.json   — camera make → file list + lens summary
 *   public/lensfun/<file>.json  — cameras + lenses from each source XML
 *
 * Usage: npx tsx scripts/convert-lensfun-db.ts
 */

import { execSync } from "child_process";
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from "fs";
import { join, basename } from "path";
import { XMLParser } from "fast-xml-parser";

const LENSFUN_REPO = "https://github.com/lensfun/lensfun.git";
const LENSFUN_DIR = join(import.meta.dirname, "..", ".lensfun-db");
const DB_DIR = join(LENSFUN_DIR, "data", "db");
const OUT_DIR = join(import.meta.dirname, "..", "public", "lensfun");

// ── Clone or update ────────────────────────────────────────────────

function ensureRepo() {
  if (!existsSync(DB_DIR)) {
    console.log("Cloning LensFun database (shallow)...");
    execSync(
      `git clone --depth 1 --filter=blob:none --sparse "${LENSFUN_REPO}" "${LENSFUN_DIR}"`,
      { stdio: "inherit" },
    );
    execSync("git sparse-checkout set data/db", {
      cwd: LENSFUN_DIR,
      stdio: "inherit",
    });
  } else {
    console.log("LensFun repo already present, pulling latest...");
    execSync("git pull --ff-only", { cwd: LENSFUN_DIR, stdio: "inherit" });
  }
}

// ── XML Parsing ────────────────────────────────────────────────────

const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "",
  // Always wrap these in arrays even when there's only one element
  isArray: (_name, jpath) => {
    const arrayPaths = [
      "lensdatabase.mount",
      "lensdatabase.camera",
      "lensdatabase.lens",
      "lensdatabase.lens.mount",
      "lensdatabase.lens.model",
      "lensdatabase.lens.calibration.distortion",
      "lensdatabase.lens.calibration.tca",
      "lensdatabase.lens.calibration.vignetting",
      "lensdatabase.lens.calibration.crop",
      "mount.compat",
    ];
    return arrayPaths.includes(jpath);
  },
});

// ── Type definitions ───────────────────────────────────────────────

interface Camera {
  make: string;
  model: string;
  mount: string;
  cropfactor: number;
  variant?: string;
}

interface Distortion {
  model: string;
  focal: number;
  k1?: number;
  k2?: number;
  a?: number;
  b?: number;
  c?: number;
}

interface TCA {
  model: string;
  focal: number;
  vr?: number;
  vb?: number;
  br?: number;
  cr?: number;
  bb?: number;
  cb?: number;
  kr?: number;
  kb?: number;
}

interface Vignetting {
  model: string;
  focal: number;
  aperture: number;
  distance: number;
  k1: number;
  k2: number;
  k3: number;
}

interface Lens {
  make: string;
  model: string;
  aliases: string[];
  mounts: string[];
  cropfactor: number;
  type?: string;
  distortion: Distortion[];
  tca: TCA[];
  vignetting: Vignetting[];
}

interface DbFile {
  cameras: Camera[];
  lenses: Lens[];
}

interface IndexEntry {
  file: string;
  cameras: string[];
  lenses: string[];
}

// ── Conversion ─────────────────────────────────────────────────────

function parseFloat_(v: unknown): number {
  return typeof v === "number" ? v : parseFloat(String(v));
}

function extractText(field: unknown): string {
  if (typeof field === "string") return field;
  if (typeof field === "number") return String(field);
  if (!Array.isArray(field)) {
    // Object with lang attr — extract #text
    const obj = field as Record<string, unknown>;
    return String(obj["#text"] ?? "");
  }
  // Array: pick the first entry without a lang attr, or just the first
  for (const m of field) {
    if (typeof m === "string") return m;
    const obj = m as Record<string, unknown>;
    if (!obj["lang"]) return String(obj["#text"] ?? "");
  }
  const first = field[0];
  if (typeof first === "string") return first;
  return String((first as Record<string, unknown>)["#text"] ?? "");
}

function extractModel(modelField: unknown): { primary: string; aliases: string[] } {
  if (!Array.isArray(modelField)) {
    return { primary: extractText(modelField), aliases: [] };
  }
  let primary = "";
  const aliases: string[] = [];
  for (const m of modelField) {
    const text = typeof m === "string" ? m : String((m as Record<string, unknown>)["#text"] ?? "");
    const lang = typeof m === "object" ? (m as Record<string, unknown>)["lang"] : undefined;
    if (!lang && !primary) primary = text;
    else aliases.push(text);
  }
  return { primary, aliases };
}

function convertCamera(raw: Record<string, unknown>): Camera {
  return {
    make: extractText(raw["maker"]),
    model: extractText(raw["model"]),
    mount: extractText(raw["mount"]),
    cropfactor: parseFloat_(raw["cropfactor"] ?? 1),
    ...(raw["variant"] ? { variant: extractText(raw["variant"]) } : {}),
  };
}

function convertDistortion(raw: Record<string, unknown>): Distortion {
  const d: Distortion = {
    model: String(raw["model"]),
    focal: parseFloat_(raw["focal"]),
  };
  if (raw["k1"] != null) d.k1 = parseFloat_(raw["k1"]);
  if (raw["k2"] != null) d.k2 = parseFloat_(raw["k2"]);
  if (raw["a"] != null) d.a = parseFloat_(raw["a"]);
  if (raw["b"] != null) d.b = parseFloat_(raw["b"]);
  if (raw["c"] != null) d.c = parseFloat_(raw["c"]);
  return d;
}

function convertTCA(raw: Record<string, unknown>): TCA {
  const t: TCA = {
    model: String(raw["model"]),
    focal: parseFloat_(raw["focal"]),
  };
  for (const k of ["vr", "vb", "br", "cr", "bb", "cb", "kr", "kb"] as const) {
    if (raw[k] != null) (t as Record<string, unknown>)[k] = parseFloat_(raw[k]);
  }
  return t;
}

function convertVignetting(raw: Record<string, unknown>): Vignetting {
  return {
    model: String(raw["model"]),
    focal: parseFloat_(raw["focal"]),
    aperture: parseFloat_(raw["aperture"]),
    distance: parseFloat_(raw["distance"]),
    k1: parseFloat_(raw["k1"]),
    k2: parseFloat_(raw["k2"]),
    k3: parseFloat_(raw["k3"]),
  };
}

function convertLens(raw: Record<string, unknown>): Lens {
  const { primary, aliases } = extractModel(raw["model"]);
  const cal = (raw["calibration"] ?? {}) as Record<string, unknown>;

  const mountRaw = raw["mount"];
  const mounts: string[] = Array.isArray(mountRaw)
    ? mountRaw.map((m: unknown) => extractText(m))
    : mountRaw
      ? [extractText(mountRaw)]
      : [];

  return {
    make: extractText(raw["maker"]),
    model: primary,
    aliases,
    mounts,
    cropfactor: parseFloat_(raw["cropfactor"] ?? 1),
    ...(raw["type"] ? { type: String(raw["type"]) } : {}),
    distortion: ((cal["distortion"] as unknown[]) ?? []).map((d) =>
      convertDistortion(d as Record<string, unknown>),
    ),
    tca: ((cal["tca"] as unknown[]) ?? []).map((t) =>
      convertTCA(t as Record<string, unknown>),
    ),
    vignetting: ((cal["vignetting"] as unknown[]) ?? []).map((v) =>
      convertVignetting(v as Record<string, unknown>),
    ),
  };
}

function convertFile(xmlPath: string): DbFile {
  const xml = readFileSync(xmlPath, "utf-8");
  const parsed = parser.parse(xml);
  const db = parsed["lensdatabase"] ?? {};

  const cameras: Camera[] = ((db["camera"] as unknown[]) ?? []).map((c) =>
    convertCamera(c as Record<string, unknown>),
  );
  const lenses: Lens[] = ((db["lens"] as unknown[]) ?? []).map((l) =>
    convertLens(l as Record<string, unknown>),
  );

  return { cameras, lenses };
}

// ── Main ───────────────────────────────────────────────────────────

function main() {
  ensureRepo();
  mkdirSync(OUT_DIR, { recursive: true });

  const xmlFiles = readdirSync(DB_DIR).filter((f) => f.endsWith(".xml"));
  const index: IndexEntry[] = [];
  let totalCameras = 0;
  let totalLenses = 0;

  for (const xmlFile of xmlFiles) {
    const xmlPath = join(DB_DIR, xmlFile);
    const result = convertFile(xmlPath);

    if (result.cameras.length === 0 && result.lenses.length === 0) continue;

    const jsonName = xmlFile.replace(".xml", ".json");
    const outPath = join(OUT_DIR, jsonName);
    writeFileSync(outPath, JSON.stringify(result));

    index.push({
      file: jsonName,
      cameras: result.cameras.map((c) => `${c.make}|${c.model}`),
      lenses: result.lenses.map((l) => `${l.make}|${l.model}`),
    });

    totalCameras += result.cameras.length;
    totalLenses += result.lenses.length;

    console.log(
      `  ${jsonName}: ${result.cameras.length} cameras, ${result.lenses.length} lenses`,
    );
  }

  const indexPath = join(OUT_DIR, "index.json");
  writeFileSync(indexPath, JSON.stringify(index));

  console.log(`\nDone: ${totalCameras} cameras, ${totalLenses} lenses across ${index.length} files`);
  console.log(`Output: ${OUT_DIR}`);
}

main();
