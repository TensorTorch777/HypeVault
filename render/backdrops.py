"""Generate cinematic dusk backdrops with Stable Diffusion XL (local, RTX GPU).

These are scenery only (no people, no text, no logos). The site labels them as
AI-generated; the watches in front of them are Blender renders.

    .venv/bin/python render/backdrops.py                 # all scenes, 2 variants each
    .venv/bin/python render/backdrops.py --only bay_dusk --variants 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "renders" / "backdrops"
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

STYLE = (
    "cinematic still, photorealistic, shot on 35mm film, anamorphic, soft film grain, "
    "pastel dusk palette of peach, pink and lavender, high detail, natural light"
)
NEGATIVE = (
    "people, person, face, hands, text, letters, watermark, logo, signature, cartoon, illustration, "
    "painting, 3d render, lowres, blurry, oversaturated, deformed, frame, border"
)

SCENES = {
    "bay_dusk": "aerial view across a bay toward a coastal art deco city skyline at dusk, lights coming on, reflections on calm water, palm trees in the foreground",
    "ocean_drive": "empty ocean-front boulevard lined with pastel art deco hotels at golden hour, softly glowing neon signs, tall palm trees, warm haze",
    "marina": "luxury yachts moored in a marina at sunset, still water mirroring a pink and lavender sky, distant towers",
    "convertible": "close view of the cream leather seat and polished chrome of a vintage convertible parked by the beach at dusk, shallow depth of field",
    "rooftop_pool": "rooftop infinity pool at blue hour with a white marble edge, overlooking glittering city lights and palm silhouettes",
    "neon_rain": "rain-soaked night street with pink and soft cyan neon reflections on wet asphalt, art deco facades, moody atmosphere",
    "penthouse": "luxury penthouse at dusk, black marble table in the foreground, floor-to-ceiling windows over a coastal skyline, warm lamps",
    "beach_dawn": "quiet beach at dawn with a pastel lifeguard tower, gentle waves, pink sky, mist over the sea",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--seed", type=int, default=1919)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.to("cuda")
    pipe.enable_vae_slicing()

    names = args.only or list(SCENES)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in names:
        for k in range(args.variants):
            seed = args.seed + 101 * k + sum(map(ord, name))
            g = torch.Generator("cuda").manual_seed(seed)
            image = pipe(
                prompt=f"{SCENES[name]}, {STYLE}",
                negative_prompt=NEGATIVE,
                num_inference_steps=args.steps,
                guidance_scale=6.0,
                width=args.width,
                height=args.height,
                generator=g,
            ).images[0]
            path = OUT / f"{name}_{k}.webp"
            image.save(path, quality=90)
            manifest[f"{name}_{k}"] = {"prompt": SCENES[name], "seed": seed, "model": MODEL, "generated": True}
            print(f"[backdrop] {path.name}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
