"""Bulk Chrono24 collection for luxury watches: listing data (CSV + JSONL) and, optionally, photos.

Listings go to local_data/chrono24/bulk/ (listings.jsonl + listings.csv).
With --download-images, listing photos go through the same research-only image pipeline as the other
scrapers (local_data/scraper_artifacts/unverified_web_images/watches/<brand>/) and are labelled
UNVERIFIED_WEB_IMAGE. A Chrono24 listing photo is not evidence that the watch is authentic.

Providers: apify (default; needs APIFY_TOKEN) or playwright (local headless browser; honours robots.txt).
Importing this module does nothing. Network access needs --allow-network.

Examples:
  python scraper/chrono24_scraper.py --dry-run
  APIFY_TOKEN=... python scraper/chrono24_scraper.py --allow-network --per-brand 200
  APIFY_TOKEN=... python scraper/chrono24_scraper.py --allow-network --brands "Patek Philippe,Rolex" --download-images
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_import_paths() -> None:
    repo = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != script_dir]
    for p in (str(repo), str(repo / "backend")):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_import_paths()

import argparse
import asyncio
import logging
import os

LOGGER = logging.getLogger("hypevault.scraper.chrono24")

# The five brands HypeVault screens, then the wider luxury market.
SUPPORTED_BRANDS = [
    "Patek Philippe",
    "Audemars Piguet",
    "Vacheron Constantin",
    "A. Lange & Söhne",
    "Richard Mille",
]
LUXURY_BRANDS = SUPPORTED_BRANDS + [
    "Rolex",
    "Omega",
    "Cartier",
    "Jaeger-LeCoultre",
    "IWC",
    "Breguet",
    "Blancpain",
    "Hublot",
    "Panerai",
    "Tudor",
    "Breitling",
    "Zenith",
    "Grand Seiko",
    "F.P. Journe",
    "H. Moser & Cie",
    "Glashütte Original",
    "Chopard",
    "Piaget",
    "Ulysse Nardin",
    "Girard-Perregaux",
    "Bvlgari",
    "TAG Heuer",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brands", default="", help="Comma-separated brands. Default: all luxury brands below.")
    parser.add_argument("--supported-only", action="store_true", help="Only the five brands HypeVault screens.")
    parser.add_argument("--queries", default="", help="Comma-separated extra free-text queries (e.g. 'Nautilus 5711').")
    parser.add_argument("--per-brand", type=int, default=100, help="Listings per brand/query (Apify max 10,000).")
    parser.add_argument("--provider", choices=("apify", "playwright"), default="apify")
    parser.add_argument("--apify-actor", default=os.environ.get("CHRONO24_APIFY_ACTOR", "epicscrapers~chrono24-scraper"))
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--no-details", action="store_true", help="Skip detail enrichment (cheaper; no ref/condition/year).")
    parser.add_argument("--out-dir", default="local_data/chrono24/bulk")
    parser.add_argument("--download-images", action="store_true", help="Also store listing photos in the research image pipeline.")
    parser.add_argument("--images-per-listing", type=int, default=1)
    parser.add_argument("--artifact-root", default=None, help="Override local_data/scraper_artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only: no network, no writes.")
    parser.add_argument("--allow-network", action="store_true", help="Required to contact Apify / Chrono24.")
    return parser.parse_args(argv)


def plan(args: argparse.Namespace) -> list[tuple[str | None, str]]:
    """(brand filter, query) pairs to run."""
    if args.brands:
        brands = [b.strip() for b in args.brands.split(",") if b.strip()]
    else:
        brands = SUPPORTED_BRANDS if args.supported_only else LUXURY_BRANDS
    jobs: list[tuple[str | None, str]] = [(b, b) for b in brands]
    jobs += [(None, q.strip()) for q in args.queries.split(",") if q.strip()]
    return jobs


def _make_provider(args: argparse.Namespace):
    from market.chrono24_live import ApifyChrono24Provider, PlaywrightChrono24Provider

    if args.provider == "apify":
        return ApifyChrono24Provider(
            os.environ.get("APIFY_TOKEN", ""),
            actor=args.apify_actor,
            currency=args.currency,
            enrich_details=not args.no_details,
            timeout_sec=max(120.0, args.per_brand * 1.5),
        )
    return PlaywrightChrono24Provider(os.environ.get("CHRONO24_SEARCH_BASE", "https://www.chrono24.com"))


def _store_images(section: dict, brand: str, args: argparse.Namespace) -> int:
    import requests

    from hypevault_scraper.common import (
        ARTIFACT_ROOT_REL,
        STATUS_UNVERIFIED_WEB,
        StoreStats,
        ValidationError,
        download_bounded,
        family_output_dir,
        load_corpus_index,
        resolve_under_repo,
        store_validated_image,
    )

    artifact_root = resolve_under_repo(args.artifact_root, ARTIFACT_ROOT_REL)
    index = load_corpus_index(artifact_root)
    stats = StoreStats()
    stored = 0
    with requests.Session() as http:
        http.headers["User-Agent"] = "HypeVault research prototype"
        for item in section["items"]:
            item_brand = item.get("brand") or brand
            dest = family_output_dir("unverified_web", "watches", item_brand, root=artifact_root.parents[1])
            for url in item.get("image_urls", [])[: args.images_per_listing]:
                try:
                    data, content_type = download_bounded(http, url)
                except (ValidationError, requests.RequestException) as exc:
                    LOGGER.info("image skipped %s: %s", url, exc)
                    continue
                record = store_validated_image(
                    data,
                    dest_dir=dest,
                    artifact_root=artifact_root,
                    family="watches",
                    claimed_brand=item_brand,
                    claimed_model=item.get("model"),
                    label_status=STATUS_UNVERIFIED_WEB,
                    search_query=section["query"],
                    search_provider=f"chrono24:{section['provider']}",
                    source_image_url=url,
                    source_page_url=item.get("listing_url"),
                    provenance_note="Chrono24 listing photo; seller-supplied, authenticity unknown.",
                    content_type=content_type,
                    index=index,
                    stats=stats,
                )
                stored += int(record is not None)
    index.save()
    return stored


async def run(args: argparse.Namespace) -> int:
    from config import REPO_ROOT
    from market.chrono24_live import Chrono24Error, search_live

    provider = _make_provider(args)
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    total = 0
    for brand, query in plan(args):
        try:
            section = await search_live(provider, query, brand=brand, limit=args.per_brand, export_dir=out_dir)
        except Chrono24Error as exc:
            LOGGER.error("%s: %s", query, exc)
            continue
        total += section["count"]
        msg = f"{query}: {section['count']} listings"
        if args.download_images and section["items"]:
            msg += f", {_store_images(section, brand or query, args)} photos stored"
        LOGGER.info(msg)
        # Be gentle with the source between queries.
        await asyncio.sleep(2 if args.provider == "apify" else 8)
    LOGGER.info("done: %d listings -> %s", total, out_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    jobs = plan(args)
    if args.dry_run:
        for brand, query in jobs:
            print(f"[plan] provider={args.provider} query={query!r} brand_filter={brand!r} limit={args.per_brand}")
        print(f"[plan] {len(jobs)} searches -> {args.out_dir}/listings.{{csv,jsonl}}" + (" + photos" if args.download_images else ""))
        return 0
    if not args.allow_network:
        raise SystemExit("Refusing to contact external services. Re-run with --allow-network (or --dry-run to plan).")
    if args.provider == "apify" and not os.environ.get("APIFY_TOKEN"):
        raise SystemExit("APIFY_TOKEN is not set. Get one at https://console.apify.com/account/integrations.")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
