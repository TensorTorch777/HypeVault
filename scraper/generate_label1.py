import torch
import os
import time
import random
import hashlib
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image

output_dir = "Label_1_Watches"
TARGET_PER_CLASS = 3000
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 1024
SAVE_SIZE = 512

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


def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def build_prompts(brand, models):
    prompts = []
    for model in models:
        for template in STYLE_TEMPLATES:
            prompts.append(template.format(brand=brand, model=model))
    random.shuffle(prompts)
    return prompts


def main():
    print("Loading SDXL pipeline (this may take a minute on first run)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_attention_slicing()

    print("Pipeline loaded. Starting generation...\n")

    os.makedirs(output_dir, exist_ok=True)
    generator = torch.Generator(device="cuda")
    total_generated = 0
    start_time = time.time()

    for brand, models in BRANDS.items():
        safe_brand = brand.replace("/", "_")
        brand_folder = os.path.join(output_dir, safe_brand)
        os.makedirs(brand_folder, exist_ok=True)

        current = count_images(brand_folder)
        needed = TARGET_PER_CLASS - current

        if needed <= 0:
            print(f"[{brand}] Already at {current}/{TARGET_PER_CLASS}. Skipping.")
            continue

        prompts = build_prompts(brand, models)

        print(f"{'='*60}")
        print(f"[{brand}] Current: {current} | Target: {TARGET_PER_CLASS} | Need: {needed}")
        print(f"[{brand}] {len(prompts)} prompt variations")
        print(f"{'='*60}")

        img_idx = 0
        prompt_idx = 0

        while count_images(brand_folder) < TARGET_PER_CLASS:
            prompt = prompts[prompt_idx % len(prompts)]
            seed = random.randint(0, 2**32 - 1)
            generator.manual_seed(seed)

            try:
                image = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    generator=generator,
                ).images[0]

                image_resized = image.resize((SAVE_SIZE, SAVE_SIZE), Image.LANCZOS)
                file_hash = hashlib.md5(f"{prompt}_{seed}".encode()).hexdigest()[:12]
                file_path = os.path.join(brand_folder, f"{file_hash}.jpg")
                image_resized.save(file_path, "JPEG", quality=95)

                img_idx += 1
                total_generated += 1
                current_count = count_images(brand_folder)
                elapsed = time.time() - start_time
                rate = total_generated / elapsed if elapsed > 0 else 0
                remaining_total = sum(
                    max(0, TARGET_PER_CLASS - count_images(os.path.join(output_dir, b.replace("/", "_"))))
                    for b in BRANDS
                )
                eta_m = (remaining_total / rate / 60) if rate > 0 else 0

                if img_idx % 10 == 0 or current_count >= TARGET_PER_CLASS:
                    print(f"  [{current_count}/{TARGET_PER_CLASS}] +{img_idx} images "
                          f"({rate:.2f} img/s, ETA {eta_m:.0f}m)")

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM at prompt_idx={prompt_idx}, clearing cache and retrying...")
                time.sleep(2)
                continue
            except Exception as e:
                print(f"  Error: {e}")

            prompt_idx += 1

        final = count_images(brand_folder)
        print(f"\n  [{brand}] Final: {final}/{TARGET_PER_CLASS}\n")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    for brand in BRANDS:
        safe_brand = brand.replace("/", "_")
        folder = os.path.join(output_dir, safe_brand)
        cnt = count_images(folder)
        status = "OK" if cnt >= TARGET_PER_CLASS else f"need {TARGET_PER_CLASS - cnt} more"
        print(f"  {brand:<25} {cnt:>5}/{TARGET_PER_CLASS}  {status}")
    print(f"\nTotal generated: {total_generated}")


if __name__ == "__main__":
    main()
