"""Collect and generate the fresh benchmark_v4 images (protocol.json, written before any data).

Same admission code as benchmark_v2 and v3 (512 px normalisation and the training exclusions); every
benchmark_v1, v2 and v3 listing and image is excluded too. Also builds the Phase 74 Kandinsky training
set (kandinsky22_train), which is development data and lives apart from benchmark_v4.

    .venv/bin/python ml_rtx5080/benchmark_v4_fresh/build.py real --per-brand 60
    .venv/bin/python ml_rtx5080/benchmark_v4_fresh/build.py kandinsky22_v4 --per-brand 60
    .venv/bin/python ml_rtx5080/benchmark_v4_fresh/build.py stable_cascade_v4 --per-brand 60
    .venv/bin/python ml_rtx5080/benchmark_v4_fresh/build.py kandinsky22_train --per-brand 200

No model scores anything here.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML = HERE.parent
V3 = ML / "benchmark_v3_fresh"
sys.path.insert(0, str(ML / "benchmark_v2_fresh"))

import build as v2  # noqa: E402  (sets the 512 px normalisation)
from common import BRANDS, BenchmarkDedup, TrainingIndex, read_manifest  # noqa: E402
from generate_synthetic import SCENES  # noqa: E402

DATA = HERE / "data"
MANIFEST = DATA / "manifest.jsonl"
TRAIN_DATA = ML / "experiments" / "kandinsky_v4" / "train_images"
TRAIN_MANIFEST = TRAIN_DATA / "manifest.jsonl"
V3_MANIFEST = V3 / "data" / "manifest.jsonl"
KANDINSKY = "kandinsky-community/kandinsky-2-2-decoder"
GROUPS = {
    "kandinsky22_v4": {"model_id": KANDINSKY, "size": 768, "steps": 50, "guidance": 4.0, "seed": 76_000, "new_scenes": True},
    "stable_cascade_v4": {"model_id": "stabilityai/stable-cascade", "size": 768, "steps": 20, "guidance": 4.0, "seed": 77_000, "new_scenes": False},
    "kandinsky22_train": {"model_id": KANDINSKY, "size": 768, "steps": 50, "guidance": 4.0, "seed": 78_000, "new_scenes": False},
}
# T9-T16: used only by kandinsky22_v4, never in development or training.
NEW_SCENES = {
    "T9_nightstand": "a photo of a {brand} {model} watch on a nightstand next to a lamp, warm evening light",
    "T10_caseback": "a seller photo of the back of a {brand} {model} watch lying on a microfiber cloth",
    "T11_marble": "a top-down phone photo of a {brand} {model} watch on a marble surface",
    "T12_cafe_wrist": "a close-up photo of a {brand} {model} watch on a woman's wrist in a cafe",
    "T13_shop_stand": "a {brand} {model} watch on a watch stand inside a jewellery shop",
    "T14_car_seat": "a phone photo of a {brand} {model} watch lying on a car seat",
    "T15_velvet_tray": "a {brand} {model} watch on a black velvet tray at a watch fair",
    "T16_crown_profile": "a side photo of a {brand} {model} watch showing the crown, on a desk mat",
}


def plan(group: str, per_brand: int, models: dict[str, list[str]], seed: int, new_scenes: bool) -> list[dict]:
    """Like benchmark_v1's plan; with new_scenes, scenes alternate T1, T9, T2, T10, ..."""
    scenes = dict(SCENES)
    order = list(SCENES)
    if new_scenes:
        scenes.update(NEW_SCENES)
        order = [s for pair in zip(SCENES, NEW_SCENES) for s in pair]
    rng = random.Random(f"{group}:{seed}")
    jobs = []
    for b_index, brand in enumerate(BRANDS):
        for i in range(per_brand):
            scene_id = order[i % len(order)]
            model = rng.choice(models[brand])
            jobs.append({
                "brand": brand, "model": model, "scene": scene_id, "new_scene": scene_id in NEW_SCENES,
                "prompt": scenes[scene_id].format(brand=brand, model=model),
                "seed": seed + b_index * 10_000 + i,
            })
    return jobs


def dedup_all() -> tuple[BenchmarkDedup, list]:
    """Every v1, v2 and v3 image, admitted or not, and every admitted v4 or training image is a duplicate target."""
    dedup = BenchmarkDedup()
    earlier = list(read_manifest(v2.V1_MANIFEST)) + list(read_manifest(v2.MANIFEST)) + list(read_manifest(V3_MANIFEST))
    mine = [s for m in (MANIFEST, TRAIN_MANIFEST) if m.exists() for s in read_manifest(m)]
    for s in earlier + [s for s in mine if s.excluded is None]:
        dedup.sha256s.add(s.original_sha256)
        dedup.dhashes.append(int(s.original_dhash, 16))
    return dedup, earlier + mine


def generate(group: str, per_brand: int) -> None:
    import torch

    sys.path.insert(0, str(V3))
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_v3", V3 / "build.py")
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)

    cfg = GROUPS[group]
    manifest, out_dir = (TRAIN_MANIFEST, TRAIN_DATA / group) if group == "kandinsky22_train" else (MANIFEST, DATA / group)
    dedup, _ = dedup_all()
    done = {s.sample_id for s in read_manifest(manifest)} if manifest.exists() else set()
    training = TrainingIndex.load()
    models = v2.models_from_dataset(Path.home() / "Downloads/archive/chrono24_unique_brand_plus_ref.csv")
    jobs = [j for j in plan(group, per_brand, models, cfg["seed"], cfg["new_scenes"]) if f"{group}_{j['seed']}" not in done]
    print(f"{group}: {len(jobs)} images to generate with {cfg['model_id']}", flush=True)
    if not jobs:
        return
    cascade = cfg["model_id"] == "stabilityai/stable-cascade"
    if cascade:
        pipe = v3._cascade_pipeline()
    else:
        pipe = v2.load_pipeline(cfg["model_id"], "cuda", True)
    for n, job in enumerate(jobs, 1):
        generator = torch.Generator(device="cpu").manual_seed(job["seed"])
        kwargs = dict(prompt=job["prompt"], negative_prompt=v2.NEGATIVE, height=cfg["size"], width=cfg["size"], generator=generator)
        if cascade:
            kwargs.update(prior_num_inference_steps=cfg["steps"], prior_guidance_scale=cfg["guidance"], num_inference_steps=10, guidance_scale=0.0)
        else:
            kwargs.update(num_inference_steps=cfg["steps"], guidance_scale=cfg["guidance"])
        image = pipe(**kwargs).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        sample = v2._admit(
            buf.getvalue(), sample_id=f"{group}_{job['seed']}", label="GENERATED", group=group,
            brand=job["brand"], unit_id=f"{group}:{job['scene']}", out_dir=out_dir, training=training, dedup=dedup,
            meta={**job, "negative_prompt": v2.NEGATIVE, "generator": cfg["model_id"], "steps": cfg["steps"],
                  "guidance": cfg["guidance"], "size": cfg["size"]},
        )
        v2.append_manifest(manifest, [sample])
        print(f"[{n}/{len(jobs)}] {sample.sample_id} {job['brand']} {job['scene']} -> {sample.excluded or 'admitted'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("real", *GROUPS))
    parser.add_argument("--per-brand", type=int, default=60)
    args = parser.parse_args()
    if args.what == "real":
        # Reuse v2's collector with v4's paths, group name and duplicate set.
        v2.DATA, v2.MANIFEST, v2.dedup_with_v1 = DATA, MANIFEST, dedup_all
        real_group = "real_ebay_v4"
        original = v2._admit
        v2._admit = lambda raw, **kw: original(raw, **{**kw, "group": real_group, "out_dir": DATA / real_group})
        asyncio.run(v2.collect_real(args.per_brand, max_pages=10, delay=0.5))
    else:
        generate(args.what, args.per_brand)


if __name__ == "__main__":
    main()
