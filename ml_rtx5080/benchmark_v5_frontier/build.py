"""Assemble benchmark_v5_frontier (protocol.json): fresh real eBay photos and the ChatGPT/Gemini test images.

Every image, real or generated, goes through benchmark_v1's admission code with the benchmark_v2+
settings: centre square, Lanczos to 512 px, JPEG q90, training-set and in-benchmark duplicate checks.

    .venv/bin/python ml_rtx5080/benchmark_v5_frontier/build.py real --per-brand 40
    .venv/bin/python ml_rtx5080/benchmark_v5_frontier/build.py generated

Real photos exclude every listing in benchmark_v1-v4, the identification gallery and market_listings
rows observed before today. No model scores anything here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML = HERE.parent
REPO = ML.parent
sys.path.insert(0, str(ML / "benchmark_v2_fresh"))
sys.path.insert(0, str(REPO / "backend"))

import build as v2  # noqa: E402  (sets the 512 px normalisation)
from collect_real import _hash_id  # noqa: E402
from common import BRANDS, BenchmarkDedup, TrainingIndex, append_manifest, read_manifest  # noqa: E402

DATA = HERE / "data"
MANIFEST = DATA / "manifest.jsonl"
NORMALISED = DATA / "normalised"
EARLIER = [
    ML / "benchmark_v1_real_vs_generated" / "data" / "manifest.jsonl",
    ML / "benchmark_v2_fresh" / "data" / "manifest.jsonl",
    ML / "benchmark_v3_fresh" / "data" / "manifest.jsonl",
    ML / "benchmark_v4_fresh" / "data" / "manifest.jsonl",
]
GENERATED = {"chatgpt_v5": "chatgpt_v5_manifest.jsonl", "gemini_v5": "gemini_v5_manifest.jsonl"}
BRAND_CODES = {"patek": "Patek Philippe", "ap": "Audemars Piguet", "lange": "A. Lange & Söhne",
               "rm": "Richard Mille", "vc": "Vacheron Constantin"}


def dedup_all() -> tuple[BenchmarkDedup, list]:
    """Every earlier benchmark image and every admitted v5 image so far is a duplicate target."""
    dedup = BenchmarkDedup()
    earlier = [s for m in EARLIER if m.exists() for s in read_manifest(m)]
    mine = [s for s in read_manifest(MANIFEST) if s.excluded is None] if MANIFEST.exists() else []
    for s in earlier + mine:
        dedup.sha256s.add(s.original_sha256)
        dedup.dhashes.append(int(s.original_dhash, 16))
    return dedup, earlier + (list(read_manifest(MANIFEST)) if MANIFEST.exists() else [])


def excluded_listing_units(relax_brands: tuple[str, ...] = ()) -> set[str]:
    """Hashed eBay item ids already used: the identification gallery and market_listings before today.

    Neither source was ever used to train, select or threshold a detector, so for brands in relax_brands
    (too few fresh listings on eBay) they are not excluded. Earlier benchmarks and training images always are.
    """
    import numpy as np

    units: set[str] = set()
    gallery = REPO / "backend" / "identify" / "data" / "watch_gallery.npz"
    if gallery.exists():
        with np.load(gallery, allow_pickle=False) as z:
            units |= {_hash_id(x) for x, b in zip(z["external_id"].tolist(), z["brand"].tolist()) if b not in relax_brands}

    async def from_db() -> set[str]:
        from sqlalchemy import select

        from database import AsyncSessionLocal, MarketListing, engine

        async with AsyncSessionLocal() as session:
            rows = (await session.execute(
                select(MarketListing.external_id).where(MarketListing.source == "ebay", MarketListing.observed_on < date.today(),
                                                        MarketListing.brand.not_in(relax_brands) if relax_brands else True)
            )).scalars().all()
        await engine.dispose()
        return {_hash_id(x) for x in rows}

    return units | asyncio.run(from_db())


def real(per_brand: int, relax_brands: tuple[str, ...] = ()) -> None:
    excluded = excluded_listing_units(relax_brands)
    print(f"{len(excluded)} earlier eBay listings excluded (gallery + market_listings before {date.today()})", flush=True)
    v2.DATA, v2.MANIFEST, v2.dedup_with_v1 = DATA, MANIFEST, dedup_all
    original_dedup = v2.dedup_with_v1

    def with_excluded():
        dedup, seen = original_dedup()
        # Represent already-used listings as seen units so the collector skips them.
        stub = [type("Seen", (), {"unit_id": u, "label": "REAL_PHOTO"})() for u in excluded]
        return dedup, seen + stub

    v2.dedup_with_v1 = with_excluded
    original_admit = v2._admit
    v2._admit = lambda raw, **kw: original_admit(raw, **{**kw, "group": "real_ebay_v5", "out_dir": NORMALISED / "real_ebay_v5"})
    asyncio.run(v2.collect_real(per_brand, max_pages=10, delay=0.5))


def generated() -> None:
    """Admit the user's ChatGPT and Gemini images (already copied into data/<group>/) with the same normalisation."""
    dedup, _ = dedup_all()
    training = TrainingIndex.load()
    done = {s.sample_id for s in read_manifest(MANIFEST)} if MANIFEST.exists() else set()
    for group, manifest in GENERATED.items():
        records = [json.loads(line) for line in (DATA / manifest).read_text().splitlines()]
        n = 0
        for r in records:
            sample_id = f"{group}_{r['name']}"
            if sample_id in done:
                continue
            brand = BRAND_CODES[r["name"].split("_", 1)[0]]
            sample = v2._admit(
                (DATA / group / r["file"]).read_bytes(), sample_id=sample_id, label="GENERATED", group=group, brand=brand,
                unit_id=sample_id, out_dir=NORMALISED / group, training=training, dedup=dedup,
                meta={k: r.get(k) for k in ("prompt_number", "name", "scene_type", "name_confidence", "file", "sha256")},
            )
            append_manifest(MANIFEST, [sample])
            n += sample.excluded is None
            if sample.excluded:
                print(f"  {sample_id}: excluded ({sample.excluded})", flush=True)
        print(f"{group}: admitted {n}/{len(records)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("real", "generated"))
    parser.add_argument("--per-brand", type=int, default=40)
    parser.add_argument("--relax-brand", action="append", default=[],
                        help="real: also allow listings seen only in the identification gallery or market_listings")
    args = parser.parse_args()
    real(args.per_brand, tuple(args.relax_brand)) if args.what == "real" else generated()


if __name__ == "__main__":
    main()
