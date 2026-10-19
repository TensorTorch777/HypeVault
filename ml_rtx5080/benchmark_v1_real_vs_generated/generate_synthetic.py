"""Generate GENERATED samples with a recorded generator, prompt and seed for every image.

    # primary, held-out generator (needs about 6 GB of VRAM with --cpu-offload)
    .venv/bin/python ml_rtx5080/benchmark_v1_real_vs_generated/generate_synthetic.py --group heldout_kandinsky22 --per-brand 60
    # secondary, possibly seen family
    .venv/bin/python ml_rtx5080/benchmark_v1_real_vs_generated/generate_synthetic.py --group sdxl_base10 --per-brand 30

Model names in prompts come from the Chrono24 dataset so the watches named are real references.
"""

from __future__ import annotations

import argparse
import collections
import io
import random
from datetime import date
from pathlib import Path

from common import BRANDS, DATA, BenchmarkDedup, TrainingIndex, admit, append_manifest, read_manifest

GROUPS = {
    "heldout_kandinsky22": {"model_id": "kandinsky-community/kandinsky-2-2-decoder", "size": 768, "steps": 50, "guidance": 4.0},
    "sdxl_base10": {"model_id": "stabilityai/stable-diffusion-xl-base-1.0", "size": 1024, "steps": 30, "guidance": 7.0},
}
NEGATIVE = "text, watermark, caption, cartoon, illustration, 3d render, painting"
SCENES = {
    "T1_table_phone": "a phone photo of a {brand} {model} wristwatch lying on a wooden table, natural window light, marketplace listing photo",
    "T2_wrist": "a candid photo of a {brand} {model} watch on a man's wrist, outdoor daylight",
    "T3_box_papers": "an overhead photo of a {brand} {model} watch with its box and papers on a desk",
    "T4_white_studio": "a studio product photo of a {brand} {model} watch on a plain white background",
    "T5_pouch": "a {brand} {model} watch resting on a leather watch pouch, indoor lighting, photo taken by the seller",
    "T6_dealer_case": "a dealer photo of a {brand} {model} watch in a glass display case",
    "T7_dial_macro": "a macro photo of the dial of a {brand} {model} watch, sharp focus",
    "T8_counter_flash": "a {brand} {model} watch on a granite countertop, photographed with flash",
}
DEFAULT_MODELS = {
    "A. Lange & Söhne": ["Lange 1", "Saxonia", "Datograph", "Zeitwerk", "1815"],
    "Audemars Piguet": ["Royal Oak", "Royal Oak Offshore", "Code 11.59", "Millenary", "Jules Audemars"],
    "Patek Philippe": ["Nautilus", "Aquanaut", "Calatrava", "Annual Calendar", "Complications"],
    "Richard Mille": ["RM 011", "RM 035", "RM 055", "RM 067", "RM 010"],
    "Vacheron Constantin": ["Overseas", "Patrimony", "Traditionnelle", "Fiftysix", "Historiques"],
}


def models_from_dataset(csv_path: Path | None, top: int = 8) -> dict[str, list[str]]:
    if csv_path is None or not csv_path.exists():
        return DEFAULT_MODELS
    from market.chrono24_dataset import read_items

    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for item in read_items(csv_path, date(2023, 9, 16), dataset_sha256="not-needed"):
        if item.model:
            counts[item.brand][item.model] += 1
    return {b: [m for m, _ in counts[b].most_common(top)] or DEFAULT_MODELS[b] for b in BRANDS}


def plan(group: str, per_brand: int, models: dict[str, list[str]], seed: int) -> list[dict]:
    rng = random.Random(f"{group}:{seed}")
    jobs = []
    for b_index, brand in enumerate(BRANDS):
        for i in range(per_brand):
            scene_id = list(SCENES)[i % len(SCENES)]
            model = rng.choice(models[brand])
            jobs.append({
                "brand": brand, "model": model, "scene": scene_id,
                "prompt": SCENES[scene_id].format(brand=brand, model=model),
                "seed": seed + b_index * 10_000 + i,
            })
    return jobs


def load_pipeline(model_id: str, device: str, cpu_offload: bool):
    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
    if cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GENERATED benchmark samples")
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--per-brand", type=int, default=60)
    parser.add_argument("--seed", type=int, default=56_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chrono24-csv", type=Path, default=Path.home() / "Downloads/archive/chrono24_unique_brand_plus_ref.csv")
    parser.add_argument("--model-id", help="override the generator (smoke tests only; the group becomes smoke_<group>)")
    parser.add_argument("--steps", type=int, help="override steps (smoke tests only)")
    parser.add_argument("--size", type=int, help="override size (smoke tests only)")
    args = parser.parse_args()

    cfg = dict(GROUPS[args.group])
    smoke = bool(args.model_id or args.steps or args.size)
    group = f"smoke_{args.group}" if smoke else args.group
    cfg.update({k: v for k, v in (("model_id", args.model_id), ("steps", args.steps), ("size", args.size)) if v})
    manifest = DATA / ("smoke_manifest.jsonl" if smoke else "manifest.jsonl")

    existing = list(read_manifest(manifest))
    done = {s.sample_id for s in existing}
    dedup = BenchmarkDedup()
    for s in existing:
        if s.excluded is None:
            dedup.sha256s.add(s.original_sha256)
            dedup.dhashes.append(int(s.original_dhash, 16))
    training = TrainingIndex.load()
    jobs = plan(args.group, args.per_brand, models_from_dataset(args.chrono24_csv), args.seed)
    jobs = [j for j in jobs if f"{group}_{j['seed']}" not in done]
    print(f"{group}: {len(jobs)} images to generate with {cfg['model_id']}")
    if not jobs:
        return

    import torch

    pipe = load_pipeline(cfg["model_id"], args.device, args.cpu_offload)
    revision = str(getattr(pipe, "name_or_path", None) or pipe.config.get("_name_or_path", cfg["model_id"]))
    for n, job in enumerate(jobs, 1):
        generator = torch.Generator(device="cpu").manual_seed(job["seed"])
        image = pipe(
            prompt=job["prompt"], negative_prompt=NEGATIVE, num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"], height=cfg["size"], width=cfg["size"], generator=generator,
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        sample = admit(
            buf.getvalue(), sample_id=f"{group}_{job['seed']}", label="GENERATED", group=group,
            brand=job["brand"], unit_id=f"{group}:{job['scene']}", out_dir=DATA / group,
            training=training, dedup=dedup,
            meta={**job, "negative_prompt": NEGATIVE, "generator": cfg["model_id"], "revision": revision,
                  "steps": cfg["steps"], "guidance": cfg["guidance"], "size": cfg["size"]},
        )
        append_manifest(manifest, [sample])
        print(f"[{n}/{len(jobs)}] {sample.sample_id} {job['brand']} / {job['scene']} -> {sample.excluded or 'admitted'}", flush=True)


if __name__ == "__main__":
    main()
