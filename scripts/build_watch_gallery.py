"""Build the watch-identification gallery from eBay listing photos (five supported brands).

For every model name in the Chrono24 catalog, search the eBay Browse API (official API, free
quota) and add the listings already saved in market_listings. Each listing photo is downloaded
into memory, turned into a DINOv3 fingerprint and dropped; only the vector, the name parsed from
the title and the listing link are kept. New eBay observations are also saved to market_listings.

    .venv/bin/python scripts/build_watch_gallery.py              # about 100 API calls
    .venv/bin/python scripts/build_watch_gallery.py --no-api     # saved listings only
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import io
import re
import sys
import time
from datetime import date
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import select  # noqa: E402

from config import settings  # noqa: E402
from database import AsyncSessionLocal, MarketListing, engine  # noqa: E402
from identify.catalog import load_catalog  # noqa: E402
from identify.embedder import Embedder  # noqa: E402
from identify.gallery import Gallery  # noqa: E402
from market.repository import save_observations  # noqa: E402
from market.service import ebay_configured, make_ebay_client  # noqa: E402
from market.types import SOURCE_EBAY  # noqa: E402

IMAGE_SIDE = 500  # eBay serves resized copies: .../s-l500.jpg
PER_MODEL = 50
CONCURRENCY = 8
BATCH = 32
CHUNK = 256


def gallery_path() -> Path:
    p = Path(settings.identify_gallery_path)
    return p if p.is_absolute() else ROOT / p


def sized(url: str) -> str:
    return re.sub(r"/s-l\d+\.", f"/s-l{IMAGE_SIDE}.", url)


async def collect(use_api: bool) -> tuple[dict[str, dict], object]:
    async with AsyncSessionLocal() as session:
        catalog = await load_catalog(session)
        rows = (await session.execute(
            select(MarketListing).where(MarketListing.source == SOURCE_EBAY, MarketListing.image_url.is_not(None))
            .order_by(MarketListing.observed_on)
        )).scalars()
        items = {r.external_id: {"brand": r.brand, "title": r.title, "image_url": r.image_url, "item_url": r.item_url}
                 for r in rows}
        print(f"{len(items)} saved eBay listings with a photo")
        if use_api:
            if not ebay_configured():
                sys.exit("EBAY_CLIENT_ID / EBAY_CLIENT_SECRET are not set; use --no-api")
            client = make_ebay_client()
            searches = [(b, m) for b, models in catalog.models.items() for _k, m in models]
            searches += [(b, b) for b in catalog.brands]  # brand-wide search catches unnamed models
            new = 0
            try:
                for i, (brand, query) in enumerate(searches, 1):
                    try:
                        found, _total = await client.search(query, brand=brand, limit=PER_MODEL)
                    except Exception as exc:
                        print(f"  search {query!r} failed: {exc}")
                        continue
                    await save_observations(session, found)
                    for it in found:
                        if it.image_url and it.external_id not in items:
                            new += 1
                        if it.image_url:
                            items[it.external_id] = {"brand": it.brand, "title": it.title,
                                                     "image_url": it.image_url, "item_url": it.item_url}
                    if i % 20 == 0:
                        print(f"  {i}/{len(searches)} searches, {new} new listings")
                    await asyncio.sleep(0.2)
            finally:
                await client.aclose()
            print(f"{len(searches)} eBay searches added {new} listings")
    await engine.dispose()
    return items, catalog


async def fetch(http: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> Image.Image | None:
    async with sem:
        try:
            resp = await http.get(sized(url))
            resp.raise_for_status()
            im = Image.open(io.BytesIO(resp.content)).convert("RGB")
            im.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
            return im
        except Exception:
            return None


async def build(items: dict[str, dict], catalog) -> dict[str, Gallery]:
    """Download, fingerprint and drop the photos a chunk at a time."""
    embedder = Embedder()
    sem = asyncio.Semaphore(CONCURRENCY)
    vectors: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    kept: list[str] = []
    ids = list(items)
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http:
        for start in range(0, len(ids), CHUNK):
            chunk = ids[start:start + CHUNK]
            photos = await asyncio.gather(*(fetch(http, sem, items[k]["image_url"]) for k in chunk))
            pairs = [(k, im) for k, im in zip(chunk, photos) if im is not None]
            for b in range(0, len(pairs), BATCH):
                part = pairs[b:b + BATCH]
                for pool, vec in (await asyncio.to_thread(embedder.embed_pools, [im for _k, im in part])).items():
                    vectors[pool].append(vec)
                kept += [k for k, _im in part]
            print(f"  {min(start + CHUNK, len(ids))}/{len(ids)} listings, {len(kept)} photos fingerprinted")
    meta = {f: [] for f in ("external_id", "brand", "model", "reference", "title", "item_url")}
    for k in kept:
        it = items[k]
        name = catalog.label_title(it["title"], it["brand"])
        for f, v in (("external_id", k), ("brand", it["brand"]), ("model", name.model), ("reference", name.reference),
                     ("title", it["title"]), ("item_url", it["item_url"])):
            meta[f].append(v)
    built = date.today().isoformat()
    return {pool: Gallery(np.concatenate(v), meta, pool=pool, built_on=built) for pool, v in vectors.items()}


async def run(use_api: bool) -> None:
    started = time.time()
    items, catalog = await collect(use_api)
    galleries = await build(items, catalog)
    out = gallery_path()
    for pool, g in galleries.items():
        path = out if pool == "cls" else out.with_name(f"{out.stem}_{pool}{out.suffix}")
        g.save(path)
        print(f"saved {pool} gallery: {len(g)} photos -> {path.relative_to(ROOT)}")
    g = galleries["cls"]
    named = sum(1 for m in g.meta["model"] if m)
    by_brand = collections.Counter(g.meta["brand"])
    print(f"{named}/{len(g)} photos have a model name; per brand: {dict(by_brand)}")
    print(f"done in {time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-api", action="store_true", help="Use saved eBay listings only")
    args = parser.parse_args()
    asyncio.run(run(not args.no_api))


if __name__ == "__main__":
    main()
