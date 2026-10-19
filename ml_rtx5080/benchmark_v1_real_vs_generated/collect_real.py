"""Collect REAL_PHOTO samples: the primary image of eBay listings, through the official Browse API.

Needs EBAY_CLIENT_ID / EBAY_CLIENT_SECRET in the environment or .env. Applies the protocol's
inclusion and exclusion rules, normalises every image and appends to data/manifest.jsonl.

    .venv/bin/python ml_rtx5080/benchmark_v1_real_vs_generated/collect_real.py --per-brand 60
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re

import httpx

from common import BRANDS, DATA, BenchmarkDedup, TrainingIndex, admit, append_manifest, read_manifest

from config import settings  # noqa: E402  (backend settings; path set up by common)
from market.ebay_browse import EbayBrowseClient  # noqa: E402

MANIFEST = DATA / "manifest.jsonl"
USER_AGENT = "HypeVault-research/1.0 (academic evaluation; images are not redistributed)"
_SIZE_RE = re.compile(r"/s-l\d+\.(jpg|jpeg|png|webp)$", re.IGNORECASE)


def largest_image_url(url: str) -> str:
    """eBay serves sizes by path suffix; ask for the 1600 px rendition."""
    return _SIZE_RE.sub("/s-l1600.jpg", url)


def _hash_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


async def _download(http: httpx.AsyncClient, url: str) -> bytes | None:
    for candidate in dict.fromkeys([largest_image_url(url), url]):
        try:
            resp = await http.get(candidate, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                return resp.content
        except httpx.HTTPError:
            continue
    return None


async def collect(per_brand: int, max_pages: int, delay: float) -> None:
    existing = list(read_manifest(MANIFEST))
    dedup = BenchmarkDedup()
    for s in existing:
        if s.excluded is None:
            dedup.sha256s.add(s.original_sha256)
            dedup.dhashes.append(int(s.original_dhash, 16))
    seen_units = {s.unit_id for s in existing if s.label == "REAL_PHOTO"}
    training = TrainingIndex.load()
    client = EbayBrowseClient(
        settings.ebay_client_id, settings.ebay_client_secret,
        environment=settings.ebay_environment, marketplace_id=settings.ebay_marketplace_id,
        pseudonym_salt=settings.market_pseudonym_salt,
    )
    async with httpx.AsyncClient(timeout=30.0) as http:
        for brand in BRANDS:
            admitted = sum(1 for s in existing if s.label == "REAL_PHOTO" and s.brand == brand and not s.excluded)
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
                    sample = admit(
                        raw, sample_id=f"real_{unit}", label="REAL_PHOTO", group="real_ebay", brand=brand,
                        unit_id=unit, out_dir=DATA / "real_ebay", training=training, dedup=dedup,
                        meta={
                            "item_id_sha256_16": unit, "seller_ref": item.seller_ref, "title": item.title,
                            "price": item.price, "currency": item.currency, "item_url": item.item_url,
                            "observed_at": item.observed_at.isoformat() if item.observed_at else None,
                            "provenance": item.provenance,
                        },
                    )
                    append_manifest(MANIFEST, [sample])
                    admitted += sample.excluded is None
                print(f"{brand}: page {page + 1}, admitted {admitted}/{per_brand} (eBay reports {total} matches)")
    await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect REAL_PHOTO benchmark samples from eBay")
    parser.add_argument("--per-brand", type=int, default=60)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between image downloads")
    args = parser.parse_args()
    asyncio.run(collect(args.per_brand, args.max_pages, args.delay))


if __name__ == "__main__":
    main()
