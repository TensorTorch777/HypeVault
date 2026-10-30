"""Assemble benchmark_v6_fresh (protocol.json): fresh real eBay photos and the ChatGPT/Gemini test images.

Every image, real or generated, goes through benchmark_v1's admission code with the benchmark_v2+
settings: centre square, Lanczos to 512 px, JPEG q90, training-set and in-benchmark duplicate checks.

    .venv/bin/python ml_rtx5080/benchmark_v6_fresh/build.py ingest ~/Downloads
    .venv/bin/python ml_rtx5080/benchmark_v6_fresh/build.py real --per-brand 30
    .venv/bin/python ml_rtx5080/benchmark_v6_fresh/build.py generated

Real photos exclude every listing in benchmark_v1-v5, the identification gallery and market_listings
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
    ML / "benchmark_v5_frontier" / "data" / "manifest.jsonl",
]
GENERATED = {"chatgpt_v6": "chatgpt_v6_manifest.jsonl", "gemini_v6": "gemini_v6_manifest.jsonl",
             "gemini_edu_v6": "gemini_edu_v6_manifest.jsonl"}
# Deviation (protocol.json): Gemini showed web sources (sothebys.com, Chrono24) under these replies,
# so the model may have seen real listing photos. Reported with and without them.
WEB_SEARCH_SHOWN = {"gemini_edu_v6": {23, 29}}
BRAND_CODES = {"patek": "Patek Philippe", "ap": "Audemars Piguet", "lange": "A. Lange & Söhne",
               "rm": "Richard Mille", "vc": "Vacheron Constantin"}


def dedup_all() -> tuple[BenchmarkDedup, list]:
    """Every earlier benchmark image and every admitted v6 image so far is a duplicate target."""
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
    (too few fresh listings on eBay) they are not excluded. Earlier benchmarks (v1-v5) and training images always are.
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
    v2._admit = lambda raw, **kw: original_admit(raw, **{**kw, "group": "real_ebay_v6", "out_dir": NORMALISED / "real_ebay_v6"})
    asyncio.run(v2.collect_real(per_brand, max_pages=10, delay=0.5))


def ingest(src: Path) -> None:
    """Copy downloads named chatgpt_<name>.* / gemini_<name>.* / gemini_edu_<name>.* (names from prompts.tsv) into data/<group>/.

    Matching is by file name only; the images are not opened except to read the C2PA label from the
    original bytes, which is recorded for the label-check report.
    """
    import csv
    import hashlib
    import shutil

    from inference.provenance import read_ai_label

    prompts = {r["name"]: r for r in csv.DictReader((HERE / "prompts.tsv").open(), delimiter="\t")}
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    for group, app in (("chatgpt_v6", "chatgpt"), ("gemini_v6", "gemini"), ("gemini_edu_v6", "gemini_edu")):
        out = DATA / group
        out.mkdir(parents=True, exist_ok=True)
        records = []
        for f in sorted(src.iterdir()):
            stem, ext = f.stem, f.suffix.lower()
            if ext not in mime or not stem.startswith(app + "_") or (app == "gemini" and stem.startswith("gemini_edu_")):
                continue
            name = stem[len(app) + 1:]
            base = name[:-4] if name.endswith("_alt") else name
            if base not in prompts:
                print(f"  skip {f.name}: not a prompt name", flush=True)
                continue
            raw = f.read_bytes()
            label = read_ai_label(raw, mime[ext])
            shutil.copy2(f, out / f.name)
            r = prompts[base]
            records.append({"name": name, "file": f.name, "prompt_number": int(r["number"]), "scene_type": r["scene_type"],
                            "sha256": hashlib.sha256(raw).hexdigest(), "c2pa_ai_label": label is not None,
                            "c2pa_maker": label.maker if label else None,
                            "web_search_sources_shown": int(r["number"]) in WEB_SEARCH_SHOWN.get(group, set())})
        (DATA / GENERATED[group]).write_text("".join(json.dumps(r) + "\n" for r in records))
        print(f"{group}: {len(records)} images, {sum(r['c2pa_ai_label'] for r in records)} with a C2PA AI label", flush=True)


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
                meta={k: r.get(k) for k in ("prompt_number", "name", "scene_type", "file", "sha256", "c2pa_ai_label", "c2pa_maker",
                                            "web_search_sources_shown")},
            )
            append_manifest(MANIFEST, [sample])
            n += sample.excluded is None
            if sample.excluded:
                print(f"  {sample_id}: excluded ({sample.excluded})", flush=True)
        print(f"{group}: admitted {n}/{len(records)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("ingest", "real", "generated"))
    parser.add_argument("src", nargs="?", default=str(Path.home() / "Downloads"), help="ingest: folder with the downloads")
    parser.add_argument("--per-brand", type=int, default=30)
    parser.add_argument("--relax-brand", action="append", default=[],
                        help="real: also allow listings seen only in the identification gallery or market_listings")
    args = parser.parse_args()
    if args.what == "ingest":
        ingest(Path(args.src).expanduser())
    elif args.what == "real":
        real(args.per_brand, tuple(args.relax_brand))
    else:
        generated()


if __name__ == "__main__":
    main()
