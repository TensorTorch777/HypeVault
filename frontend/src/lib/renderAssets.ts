import fs from "node:fs";
import path from "node:path";

import { STUDIO_WATCHES, type StudioWatch } from "@/lib/studioWatches";

export { STUDIO_WATCHES, type StudioWatch };

export type RenderAssets = {
  quality: string | null;
  watches: Record<string, { hero: boolean; flat: boolean; turntable_frames: number }>;
  backdrops: string[];
};

const RENDERS = path.join(process.cwd(), "public", "renders");

/** What has actually been rendered, read from public/renders at build/request time. */
export function getRenderAssets(): RenderAssets {
  let quality: string | null = null;
  let watches: RenderAssets["watches"] = {};
  try {
    const manifest = JSON.parse(fs.readFileSync(path.join(RENDERS, "manifest.json"), "utf8"));
    quality = manifest.quality ?? null;
    watches = manifest.watches ?? {};
  } catch {
    // nothing rendered yet: the UI falls back to painted skies
  }
  let backdrops: string[] = [];
  try {
    backdrops = fs
      .readdirSync(path.join(RENDERS, "backdrops"))
      .filter((f) => f.endsWith(".webp"))
      .map((f) => f.replace(/\.webp$/, ""))
      .sort();
  } catch {
    backdrops = [];
  }
  return { quality, watches, backdrops };
}

export function backdropUrl(assets: RenderAssets, scene: string): string | null {
  const hit = assets.backdrops.find((b) => b.startsWith(`${scene}_`));
  return hit ? `/renders/backdrops/${hit}.webp` : null;
}

export function heroUrl(assets: RenderAssets, id: string, shot: "hero" | "flat" = "hero"): string | null {
  return assets.watches[id]?.[shot] ? `/renders/${id}/${shot}.webp` : null;
}
