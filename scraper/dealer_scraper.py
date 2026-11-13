"""Crawl pre-owned luxury-watch dealers ourselves: The 1916 Company, Watchfinder & Co., Chronext.

No third-party scraping service. robots.txt is honoured, requests to each site are spaced (3 s by default),
and a bot challenge or 403 stops that site instead of being worked around.

Output: local_data/dealers/<source>/listings.{csv,jsonl}. With --save-db, rows for the five supported brands
also go into market_listings, where the site's "From dealers" panel reads them. With --download-images,
listing photos go through the research image pipeline as UNVERIFIED_WEB_IMAGE.

Importing this module does nothing. Network access needs --allow-network.

  python scraper/dealer_scraper.py --probe --allow-network          # check one page per site, print what parsed
  python scraper/dealer_scraper.py --allow-network --per-brand 40 --save-db
  python scraper/dealer_scraper.py --allow-network --sites watchfinder --brands "Patek Philippe" --download-images
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

LOGGER = logging.getLogger("hypevault.scraper.dealers")

SUPPORTED_BRANDS = ["Patek Philippe", "Audemars Piguet", "Vacheron Constantin", "A. Lange & Söhne", "Richard Mille"]
EXTRA_BRANDS = ["Rolex", "Omega", "Cartier", "Jaeger-LeCoultre", "IWC", "Breguet", "Hublot", "Panerai", "Tudor", "F.P. Journe"]
ALL_SITES = ("the1916company", "watchfinder", "chronext")
PROBE_FIELDS = ("title", "price", "currency", "brand", "model", "reference", "year", "condition", "box", "papers",
                "case_size", "case_material", "dial", "image_url")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", default=",".join(ALL_SITES), help=f"Comma-separated: {', '.join(ALL_SITES)}")
    p.add_argument("--brands", default="", help="Comma-separated brands. Default: the five supported brands.")
    p.add_argument("--more-brands", action="store_true", help="Also crawl Rolex, Omega, Cartier, JLC, IWC, ...")
    p.add_argument("--per-brand", type=int, default=30, help="Product pages per brand per site.")
    p.add_argument("--delay", type=float, default=3.0, help="Seconds between requests to the same site (min 1).")
    p.add_argument("--include-sold", action="store_true")
    p.add_argument("--out-dir", default="local_data/dealers")
    p.add_argument("--save-db", action="store_true", help="Also write market_listings rows (needs DATABASE_URL).")
    p.add_argument("--download-images", action="store_true")
    p.add_argument("--images-per-listing", type=int, default=1)
    p.add_argument("--artifact-root", default=None)
    p.add_argument("--save-html", action="store_true", help="Keep raw product HTML (for fixing parsers).")
    p.add_argument("--probe", action="store_true", help="One product page per site; print what parsed.")
    p.add_argument("--dry-run", action="store_true", help="Plan only: no network, no writes.")
    p.add_argument("--allow-network", action="store_true")
    args = p.parse_args(argv)
    args.delay = max(1.0, args.delay)
    return args


def plan(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    unknown = [s for s in sites if s not in ALL_SITES]
    if unknown:
        raise SystemExit(f"unknown site(s): {', '.join(unknown)}. Choose from {', '.join(ALL_SITES)}.")
    if args.brands:
        brands = [b.strip() for b in args.brands.split(",") if b.strip()]
    else:
        brands = SUPPORTED_BRANDS + (EXTRA_BRANDS if args.more_brands else [])
    return sites, brands


def _store_images(listings, args: argparse.Namespace) -> int:
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
    from market.dealers.http import DEFAULT_USER_AGENT

    artifact_root = resolve_under_repo(args.artifact_root, ARTIFACT_ROOT_REL)
    index = load_corpus_index(artifact_root)
    stats = StoreStats()
    stored = 0
    with requests.Session() as http:
        http.headers["User-Agent"] = DEFAULT_USER_AGENT
        for item in listings:
            brand = item.brand or "unknown"
            dest = family_output_dir("unverified_web", "watches", brand, root=artifact_root.parents[1])
            for url in item.image_urls[: args.images_per_listing]:
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
                    claimed_brand=brand,
                    claimed_model=item.model,
                    label_status=STATUS_UNVERIFIED_WEB,
                    search_query=item.reference or item.title,
                    search_provider=f"dealer:{item.source}",
                    source_image_url=url,
                    source_page_url=item.listing_url,
                    provenance_note=f"{item.source} dealer listing photo; authenticity not established by us.",
                    content_type=content_type,
                    index=index,
                    stats=stats,
                )
                stored += int(record is not None)
    index.save()
    return stored


async def _save_db(listings) -> int:
    from database import AsyncSessionLocal, engine
    from market.dealers.crawl import to_market_items
    from market.repository import save_observations

    items = to_market_items(listings)
    written = 0
    async with AsyncSessionLocal() as session:
        for start in range(0, len(items), 500):
            written += await save_observations(session, items[start : start + 500])
    await engine.dispose()
    return written


async def probe(args: argparse.Namespace) -> int:
    from config import REPO_ROOT
    from market.dealers.extract import slug_candidates
    from market.dealers.http import BlockedError, FetchError, PoliteFetcher
    from market.dealers.sites import SITES

    sites, brands = plan(args)
    brand = brands[0]
    html_dir = REPO_ROOT / args.out_dir / "_probe"
    html_dir.mkdir(parents=True, exist_ok=True)
    async with PoliteFetcher(min_delay=args.delay) as fetcher:
        for source in sites:
            site = SITES[source]()
            print(f"\n=== {site.label} ({site.base_url}) — {brand}")
            try:
                sitemaps = await fetcher.sitemaps(site.base_url)
                print(f"robots.txt sitemaps: {len(sitemaps)} {sitemaps[:3]}")
                urls = await site.discover(fetcher, brand, 3)
                print(f"product pages found: {len(urls)} (requests so far: {fetcher.requests})")
                for u in urls:
                    print(f"  {u}")
                if not urls:
                    page1 = site.category_url(slug_candidates(brand)[0], 1)
                    html = await fetcher.get_text(page1)
                    if html:
                        (html_dir / f"{source}-category.html").write_text(html, encoding="utf-8")
                        print(f"  no product links matched; saved category page HTML to {html_dir / (source + '-category.html')}")
                    continue
                html = await fetcher.get_text(urls[0])
                if not html:
                    print("  first product page returned 404")
                    continue
                path = html_dir / f"{source}-product.html"
                path.write_text(html, encoding="utf-8")
                listing = site.parse(urls[0], html, brand)
                if listing is None:
                    print(f"  PARSE FAILED (no title or price). HTML saved to {path}")
                    continue
                d = listing.as_dict()
                ok = [f for f in PROBE_FIELDS if d.get(f) not in (None, "", [])]
                missing = [f for f in PROBE_FIELDS if f not in ok]
                for f in ok:
                    print(f"  {f:14} {str(d[f])[:90]}")
                print(f"  missing: {', '.join(missing) or 'none'}")
                print(f"  HTML saved to {path}")
            except BlockedError as exc:
                print(f"  BLOCKED: {exc}")
            except FetchError as exc:
                print(f"  ERROR: {exc}")
    return 0


async def run(args: argparse.Namespace) -> int:
    from config import REPO_ROOT
    from market.dealers.crawl import crawl_site, export_listings
    from market.dealers.http import PoliteFetcher
    from market.dealers.sites import SITES

    sites, brands = plan(args)
    out_root = REPO_ROOT / args.out_dir

    async def one(source: str):
        site = SITES[source]()
        # One fetcher per site so each site keeps its own request spacing; sites run in parallel.
        async with PoliteFetcher(min_delay=args.delay) as fetcher:
            report = await crawl_site(
                site, fetcher, brands, args.per_brand, include_sold=args.include_sold,
                save_html_dir=(out_root / source / "html") if args.save_html else None,
            )
        export_listings(out_root / source, report.listings)
        return site, report

    results = await asyncio.gather(*(one(s) for s in sites))
    all_listings = []
    for site, report in results:
        all_listings += report.listings
        line = (f"{site.label}: {len(report.listings)} listings saved "
                f"({report.discovered} found, {report.pages_fetched} fetched, {report.skipped} skipped)")
        if report.blocked:
            line += f" — STOPPED: {report.blocked}"
        if report.errors:
            line += f" — {len(report.errors)} errors (first: {report.errors[0]})"
        LOGGER.info(line)
    if args.save_db and all_listings:
        LOGGER.info("market_listings: %d new rows", await _save_db(all_listings))
    if args.download_images and all_listings:
        LOGGER.info("photos stored: %d", _store_images(all_listings, args))
    LOGGER.info("output: %s", out_root)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args(argv)
    sites, brands = plan(args)
    if args.dry_run:
        for s in sites:
            print(f"[plan] {s}: {len(brands)} brands x up to {args.per_brand} pages, {args.delay:.0f}s apart -> {args.out_dir}/{s}/")
        print(f"[plan] brands: {', '.join(brands)}" + (" | +db" if args.save_db else "") + (" | +photos" if args.download_images else ""))
        return 0
    if not args.allow_network:
        raise SystemExit("Refusing to contact external sites. Re-run with --allow-network (or --dry-run to plan).")
    return asyncio.run(probe(args) if args.probe else run(args))


if __name__ == "__main__":
    raise SystemExit(main())
