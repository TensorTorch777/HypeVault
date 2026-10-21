"""Collect and generate the fresh benchmark_v2 images (protocol.json, written before any data).

Reuses benchmark_v1's admission code (normalisation, training and in-benchmark exclusions) with the
v2 changes: 512 px normalisation, new seeds, aMUSEd as a second generator, and every benchmark_v1
listing and image excluded.

    .venv/bin/python ml_rtx5080/benchmark_v2_fresh/build.py real --per-brand 60
    .venv/bin/python ml_rtx5080/benchmark_v2_fresh/build.py kandinsky22_fresh --per-brand 60
    .venv/bin/python ml_rtx5080/benchmark_v2_fresh/build.py amused512 --per-brand 60

No model scores anything here.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "benchmark_v1_real_vs_generated"
sys.path.insert(0, str(V1))

import common  # noqa: E402

common.NORMALISED_SIDE = 512
common.MIN_SOURCE_SIDE = 512

from collect_real import _download, _hash_id  # noqa: E402
from common import BRANDS, BenchmarkDedup, TrainingIndex, admit, append_manifest, read_manifest  # noqa: E402
from generate_synthetic import NEGATIVE, load_pipeline, models_from_dataset, plan  # noqa: E402

DATA = HERE / "data"
MANIFEST = DATA / "manifest.jsonl"
V1_MANIFEST = V1 / "data" / "manifest.jsonl"
GROUPS = {
    "kandinsky22_fresh": {"model_id": "kandinsky-community/kandinsky-2-2-decoder", "size": 768, "steps": 50, "guidance": 4.0, "seed": 72_000},
    "amused512": {"model_id": "amused/amused-512", "size": 512, "steps": 12, "guidance": 10.0, "seed": 73_000},
}


def _admit(raw: bytes, **kwargs):
    sample = admit(raw, **kwargs)
    if sample.excluded == "source_short_side_below_768":
        sample.excluded = "source_short_side_below_512"
    return sample


def dedup_with_v1() -> tuple[BenchmarkDedup, list]:
    """Every v1 image (admitted or not) and every v2 image so far is a duplicate target."""
    dedup = BenchmarkDedup()
    v1 = list(read_manifest(V1_MANIFEST))
    v2 = list(read_manifest(MANIFEST)) if MANIFEST.exists() else []
    for s in v1 + [s for s in v2 if s.excluded is None]:
        dedup.sha256s.add(s.original_sha256)
        dedup.dhashes.append(int(s.original_dhash, 16))
    return dedup, v1 + v2


async def collect_real(per_brand: int, max_pages: int, delay: float) -> None:
    import httpx

    from config import settings
    from market.ebay_browse import EbayBrowseClient

    dedup, seen = dedup_with_v1()
    v2 = list(read_manifest(MANIFEST)) if MANIFEST.exists() else []
    seen_units = {s.unit_id for s in seen if s.label == "REAL_PHOTO"}
    training = TrainingIndex.load()
    client = EbayBrowseClient(
        settings.ebay_client_id, settings.ebay_client_secret,
        environment=settings.ebay_environment, marketplace_id=settings.ebay_marketplace_id,
        pseudonym_salt=settings.market_pseudonym_salt,
    )
    async with httpx.AsyncClient(timeout=30.0) as http:
        for brand in BRANDS:
            admitted = sum(1 for s in v2 if s.label == "REAL_PHOTO" and s.brand == brand and not s.excluded)
            for page in range(max_pages):
                if admitted >= per_brand:
                    break
                items, total = await client.search(brand, brand=brand, limit=100, offset=page * 100, min_price=1000)
                if not items:
                    break
                for item in items:
                    unit = _hash_id(item.external_id)
                    if admitted >= per_brand or unit in seen_units or not item.image_url or item.price < 1000:
                        continue
                    seen_units.add(unit)
                    raw = await _download(http, item.image_url)
                    await asyncio.sleep(delay)
                    if raw is None:
                        continue
                    sample = _admit(
                        raw, sample_id=f"real_{unit}", label="REAL_PHOTO", group="real_ebay_v2", brand=brand,
                        unit_id=unit, out_dir=DATA / "real_ebay_v2", training=training, dedup=dedup,
                        meta={"item_id_sha256_16": unit, "seller_ref": item.seller_ref, "title": item.title,
                              "price": item.price, "currency": item.currency, "item_url": item.item_url,
                              "observed_at": item.observed_at.isoformat() if item.observed_at else None,
                              "provenance": item.provenance},
                    )
                    append_manifest(MANIFEST, [sample])
                    admitted += sample.excluded is None
                print(f"{brand}: page {page + 1}, admitted {admitted}/{per_brand} (eBay reports {total})", flush=True)
    await client.aclose()


def _amused_pipeline():
    import torch
    from diffusers import AmusedPipeline

    pipe = AmusedPipeline.from_pretrained("amused/amused-512", variant="fp16", torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(group: str, per_brand: int) -> None:
    import torch

    cfg = GROUPS[group]
    dedup, _ = dedup_with_v1()
    done = {s.sample_id for s in read_manifest(MANIFEST)} if MANIFEST.exists() else set()
    training = TrainingIndex.load()
    jobs = [j for j in plan(group, per_brand, models_from_dataset(Path.home() / "Downloads/archive/chrono24_unique_brand_plus_ref.csv"), cfg["seed"])
            if f"{group}_{j['seed']}" not in done]
    print(f"{group}: {len(jobs)} images to generate with {cfg['model_id']}", flush=True)
    if not jobs:
        return
    pipe = _amused_pipeline() if group == "amused512" else load_pipeline(cfg["model_id"], "cuda", True)
    for n, job in enumerate(jobs, 1):
        generator = torch.Generator(device="cpu").manual_seed(job["seed"])
        kwargs = dict(prompt=job["prompt"], negative_prompt=NEGATIVE, num_inference_steps=cfg["steps"],
                      guidance_scale=cfg["guidance"], height=cfg["size"], width=cfg["size"], generator=generator)
        image = pipe(**kwargs).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        sample = _admit(
            buf.getvalue(), sample_id=f"{group}_{job['seed']}", label="GENERATED", group=group,
            brand=job["brand"], unit_id=f"{group}:{job['scene']}", out_dir=DATA / group, training=training, dedup=dedup,
            meta={**job, "negative_prompt": NEGATIVE, "generator": cfg["model_id"], "steps": cfg["steps"],
                  "guidance": cfg["guidance"], "size": cfg["size"]},
        )
        append_manifest(MANIFEST, [sample])
        print(f"[{n}/{len(jobs)}] {sample.sample_id} {job['brand']} -> {sample.excluded or 'admitted'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("real", *GROUPS))
    parser.add_argument("--per-brand", type=int, default=60)
    args = parser.parse_args()
    if args.what == "real":
        asyncio.run(collect_real(args.per_brand, max_pages=10, delay=0.5))
    else:
        generate(args.what, args.per_brand)


if __name__ == "__main__":
    main()
