"""Generate synthetic watch imagery for research-only stress tests.

Outputs: local_data/scraper_artifacts/synthetic_research_only/watches/<brand>/.
Generated branded imagery is SYNTHETIC_AI_GENERATED, never counterfeit ground truth,
and never written into Label_1_Watches. Importing this module does not load SDXL.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _ensure_repo_import_path() -> None:
    """Prefer the repo package path over the script directory shadowing the package name."""
    repo = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != script_dir]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))



_ensure_repo_import_path()

import argparse
import logging
import random
import time

from scraper.common import (
    ARTIFACT_ROOT_REL,
    STATUS_SYNTHETIC,
    STATE_REL,
    ContentIndex,
    OOMRetryController,
    StoreStats,
    assert_not_legacy_training_dir,
    count_valid_images,
    load_resume_state,
    normalize_image_bytes,
    require_generation_opt_in,
    resolve_under_repo,
    safe_brand_name,
    save_resume_state,
    store_validated_image,
)

LOGGER = logging.getLogger("hypevault.scraper.generate_watches")

TARGET_PER_CLASS = 3000
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 1024
SAVE_SIZE = 512
MAX_OOM_RETRIES = 3
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

BRANDS = {
    "Patek Philippe": [
        "Nautilus", "Aquanaut", "Calatrava", "Grand Complications",
        "Perpetual Calendar", "World Time", "Chronograph", "Gondolo",
        "Twenty~4", "Sky Moon Tourbillon", "Celestial", "Travel Time",
        "Annual Calendar", "Minute Repeater", "Golden Ellipse",
    ],
    "Audemars Piguet": [
        "Royal Oak", "Royal Oak Offshore", "Royal Oak Offshore Diver",
        "Royal Oak Chronograph", "Royal Oak Tourbillon", "Royal Oak Concept",
        "Code 11.59", "Millenary", "Jules Audemars", "Royal Oak Perpetual Calendar",
        "Royal Oak Jumbo", "Royal Oak Day-Date", "Royal Oak Offshore Lady",
    ],
    "Vacheron Constantin": [
        "Overseas", "Patrimony", "Traditionnelle", "Fiftysix", "Historiques",
        "Harmony", "Malte", "Overseas Chronograph", "Overseas World Time",
        "Quai de l'Ile", "Royal Eagle", "Metiers d'Art",
    ],
    "A. Lange & Söhne": [
        "Lange 1", "Zeitwerk", "Saxonia", "Datograph", "1815",
        "Richard Lange", "Odysseus", "Grand Lange 1", "Cabaret",
        "Langematik", "Little Lange 1", "Grand Langematik",
    ],
    "Richard Mille": [
        "RM 011", "RM 035", "RM 055", "RM 07", "RM 67", "RM 010",
        "RM 030", "RM 027", "RM 037", "RM 052", "RM 032", "RM 005",
        "RM 016", "RM 061", "RM 63", "RM 028", "RM 029",
    ],
}

STYLE_TEMPLATES = [
    "{brand} {model} luxury watch, product photography, white background, studio lighting, high resolution, detailed",
    "{brand} {model} watch, professional product shot, black background, dramatic lighting, 8k",
    "{brand} {model} wristwatch on wrist, lifestyle photography, bokeh background, natural lighting",
    "{brand} {model} watch dial close up, macro photography, ultra detailed, sharp focus",
    "{brand} {model} timepiece, editorial photography, magazine style, clean background",
    "{brand} {model} watch, floating product shot, minimalist, soft shadow, commercial photography",
    "{brand} {model} luxury timepiece, jewelry photography, reflective surface, elegant",
    "{brand} {model} watch face, frontal view, detailed dial, high resolution photography",
    "{brand} {model} wristwatch, side angle view, showing case and crown, product catalog",
    "{brand} {model} watch, overhead shot, leather strap detail, warm lighting, premium feel",
]

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "watermark, text, logo, banner, cartoon, painting, drawing, sketch, "
    "low resolution, pixelated, noise, grainy"
)


def build_prompts(brand: str, models: list[str]) -> list[str]:
    prompts = [template.format(brand=brand, model=model) for model in models for template in STYLE_TEMPLATES]
    random.shuffle(prompts)
    return prompts


def load_pipeline():
    """Lazy-load Torch/Diffusers only when generation is explicitly allowed."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_attention_slicing()
    return pipe, torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic branded watch images for research-only directories. "
            "Outputs are SYNTHETIC_AI_GENERATED and are not counterfeit ground truth."
        )
    )
    parser.add_argument("--artifact-root", default=str(ARTIFACT_ROOT_REL))
    parser.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-generation", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifact_root = resolve_under_repo(args.artifact_root, ARTIFACT_ROOT_REL)
    state_path = resolve_under_repo(
        STATE_REL / "generate_watches_resume.json",
        STATE_REL / "generate_watches_resume.json",
    )
    failures_path = artifact_root / "state" / "generate_watches_failures.jsonl"

    if args.dry_run:
        print("DRY-RUN: no model load and no image writes.")
        print(f"Artifact root: {artifact_root}")
        print(f"Model: {MODEL_ID}")
        print(f"Brands: {', '.join(BRANDS)}")
        print("WARNING: synthetic branded imagery is not verified counterfeit evidence.")
        return 0

    require_generation_opt_in(args.allow_generation)
    pipe, torch = load_pipeline()
    generator = torch.Generator(device="cuda")
    state = load_resume_state(state_path)
    stats = StoreStats(**(state.get("stats") or {})) if isinstance(state.get("stats"), dict) else StoreStats()
    total_generated = 0
    start = time.time()

    for brand, models in BRANDS.items():
        dest_dir = artifact_root / "synthetic_research_only" / "watches" / safe_brand_name(brand)
        dest_dir.mkdir(parents=True, exist_ok=True)
        assert_not_legacy_training_dir(dest_dir)
        index = ContentIndex.load(dest_dir / "content_index.json")
        current = count_valid_images(dest_dir)
        if current >= args.target_per_class:
            print(f"[{brand}] Already at {current}/{args.target_per_class}. Skipping.")
            continue

        prompts = build_prompts(brand, models)
        print(f"[{brand}] current={current} target={args.target_per_class} prompts={len(prompts)}")
        prompt_idx = 0
        while count_valid_images(dest_dir) < args.target_per_class:
            prompt = prompts[prompt_idx % len(prompts)]
            seed = random.randint(0, 2**32 - 1)
            generator.manual_seed(seed)
            oom = OOMRetryController(max_retries=MAX_OOM_RETRIES)
            while True:
                try:
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        height=IMAGE_SIZE,
                        width=IMAGE_SIZE,
                        generator=generator,
                    ).images[0]
                    from PIL import Image as PILImage

                    image_resized = image.resize((SAVE_SIZE, SAVE_SIZE), PILImage.Resampling.LANCZOS)
                    data = normalize_image_bytes(image_resized)
                    before = stats.stored
                    store_validated_image(
                        data,
                        dest_dir=dest_dir,
                        artifact_root=artifact_root,
                        family="watches",
                        claimed_brand=brand,
                        claimed_model="UNKNOWN",
                        label_status=STATUS_SYNTHETIC,
                        search_query=None,
                        search_provider="sdxl_local",
                        source_image_url="NOT_APPLICABLE",
                        source_page_url="NOT_APPLICABLE",
                        provenance_note=(
                            "Local SDXL synthesis. Not a verified counterfeit product "
                            "and not eligible for authenticity benchmarking."
                        ),
                        generation={
                            "generator": "StableDiffusionXLPipeline",
                            "model_id": MODEL_ID,
                            "prompt": prompt,
                            "negative_prompt": NEGATIVE_PROMPT,
                            "seed": seed,
                            "num_inference_steps": args.steps,
                            "guidance_scale": args.guidance,
                            "output_resolution": f"{SAVE_SIZE}x{SAVE_SIZE}",
                            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        },
                        index=index,
                        stats=stats,
                    )
                    if stats.stored > before:
                        total_generated += 1
                    break
                except Exception as exc:  # noqa: BLE001
                    if oom.should_retry(exc):
                        LOGGER.warning("CUDA OOM retry %s/%s", oom.attempts, oom.max_retries)
                        time.sleep(2)
                        continue
                    if oom.attempts:
                        oom.record_failure(failures_path, prompt=prompt, seed=seed, error=str(exc))
                        LOGGER.error("giving up after OOM retries: %s", exc)
                    else:
                        LOGGER.error("generation error: %s", exc)
                        stats.errors += 1
                    break
            prompt_idx += 1
            state["stats"] = stats.as_dict()
            save_resume_state(state_path, state)

        final = count_valid_images(dest_dir)
        print(f"[{brand}] final={final}/{args.target_per_class} shortfall={max(0, args.target_per_class - final)}")

    print(f"DONE in {(time.time() - start) / 60:.1f}m generated={total_generated} stats={stats.as_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
