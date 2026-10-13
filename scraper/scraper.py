"""Collect unverified web watch images into research-only artifact directories.

Outputs go under local_data/scraper_artifacts/unverified_web_images/watches/.
This tool never writes Label_0_Watches / Label_1_Watches and never assigns a
verified authenticity label. Importing this module does not start a crawl.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _ensure_repo_import_path() -> None:
    """Prefer the repo package path over the script directory shadowing the package name."""
    repo = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != script_dir]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))



_ensure_repo_import_path()

import argparse
import logging
import shutil
import time

from scraper.common import (
    ARTIFACT_ROOT_REL,
    STATUS_UNVERIFIED_WEB,
    STATE_REL,
    StoreStats,
    ContentIndex,
    assert_not_legacy_training_dir,
    count_valid_images,
    load_resume_state,
    mark_query_completed,
    require_network_opt_in,
    resolve_under_repo,
    safe_brand_name,
    save_resume_state,
    store_validated_image,
)

LOGGER = logging.getLogger("hypevault.scraper.watches")

PREMIUM_BRANDS = [
    "Patek Philippe",
    "Audemars Piguet",
    "Vacheron Constantin",
    "A. Lange & Söhne",
    "Richard Mille",
]

TARGET_PER_CLASS = 3000
IMAGES_PER_QUERY = 100

SEARCH_NAMES = {
    "A. Lange & Söhne": ["A Lange Sohne", "Lange Sohne", "A. Lange Sohne"],
    "Richard Mille": ["Richard Mille"],
    "Patek Philippe": ["Patek Philippe"],
    "Audemars Piguet": ["Audemars Piguet"],
    "Vacheron Constantin": ["Vacheron Constantin"],
}

ICRAWLER_PROVENANCE_NOTE = (
    "icrawler BingImageCrawler stores downloaded files under a temp root and "
    "exposes task['file_url'] during download, but this wrapper validates files "
    "after the crawl completes. When the original URL is not recoverable from "
    "the saved filename, source_image_url is recorded as UNKNOWN."
)


def build_queries(brand: str, models: list[str], refs: list[str]) -> list[str]:
    search_names = SEARCH_NAMES.get(brand, [brand])
    queries: list[str] = []
    for name in search_names:
        for model in models:
            queries.append(f"{name} {model} watch")
            queries.append(f"{name} {model} wristwatch official photo")
            queries.append(f"{name} {model} dial close up")
        for ref in refs[:50]:
            queries.append(f"{name} {ref} watch")
        queries.extend(
            [
                f"{name} luxury watch",
                f"{name} watch collection",
                f"{name} timepiece high resolution",
                f"{name} watch face close up",
                f"{name} wristwatch product photo",
                f"{name} watch review",
                f"{name} watch catalog",
                f"{name} latest watches",
                f"{name} watch official website",
            ]
        )
    return queries


def _crawl_bing(query: str, tmp_folder: Path, max_num: int) -> None:
    from icrawler.builtin import BingImageCrawler

    if tmp_folder.exists():
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(
        storage={"root_dir": str(tmp_folder)},
        downloader_threads=4,
        log_level=logging.CRITICAL,
    )
    crawler.crawl(keyword=query, max_num=max_num, filters={"type": "photo", "size": "large"})


def ingest_temp_dir(
    tmp_folder: Path,
    *,
    dest_dir: Path,
    artifact_root: Path,
    brand: str,
    query: str,
    stats: StoreStats,
    index: ContentIndex,
) -> int:
    if not tmp_folder.exists():
        return 0
    stored_before = stats.stored
    for path in sorted(tmp_folder.iterdir()):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            stats.errors += 1
            continue
        store_validated_image(
            data,
            dest_dir=dest_dir,
            artifact_root=artifact_root,
            family="watches",
            claimed_brand=brand,
            claimed_model=None,
            label_status=STATUS_UNVERIFIED_WEB,
            search_query=query,
            search_provider="bing_icrawler",
            source_image_url="UNKNOWN",
            source_page_url="UNKNOWN",
            provenance_note=ICRAWLER_PROVENANCE_NOTE,
            index=index,
            stats=stats,
        )
    return stats.stored - stored_before


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download unverified Bing image search results for premium watch brands "
            "into local_data/scraper_artifacts/unverified_web_images/watches/. "
            "Does not assign authenticity labels."
        )
    )
    parser.add_argument(
        "--csv",
        default="Watches.csv",
        help="Catalog CSV path (repo-relative by default). Default: Watches.csv",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(ARTIFACT_ROOT_REL),
        help="Artifact root under the repository. Default: local_data/scraper_artifacts",
    )
    parser.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--images-per-query", type=int, default=IMAGES_PER_QUERY)
    parser.add_argument(
        "--temp-dir",
        default="/tmp/hypevault_watch_crawl_tmp",
        help="Temporary crawl directory (deleted after each query).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan queries; no network or writes.")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Required opt-in to contact Bing via icrawler.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifact_root = resolve_under_repo(args.artifact_root, ARTIFACT_ROOT_REL)
    assert_not_legacy_training_dir(artifact_root / "unverified_web_images")
    csv_path = resolve_under_repo(args.csv, "Watches.csv")
    temp_dir = Path(args.temp_dir)
    state_path = resolve_under_repo(STATE_REL / "watch_scraper_resume.json", STATE_REL / "watch_scraper_resume.json")

    if args.dry_run:
        print("DRY-RUN: no network and no image writes.")
        print(f"CSV: {csv_path}")
        print(f"Artifact root: {artifact_root}")
        print(f"Default output: {artifact_root / 'unverified_web_images' / 'watches'}")
        print(f"Brands: {', '.join(PREMIUM_BRANDS)}")
        print(f"Target per class: {args.target_per_class}")
        return 0

    require_network_opt_in(args.allow_network)

    import pandas as pd

    if not csv_path.exists():
        LOGGER.error("CSV not found: %s", csv_path)
        return 2

    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["brand", "ref"])
    df = df[df["brand"].isin(PREMIUM_BRANDS)]

    state = load_resume_state(state_path)
    completed = set(state.get("completed_queries") or [])
    stats = StoreStats(**(state.get("stats") or {})) if isinstance(state.get("stats"), dict) else StoreStats()
    start = time.time()

    for brand in PREMIUM_BRANDS:
        dest_dir = artifact_root / "unverified_web_images" / "watches" / safe_brand_name(brand)
        dest_dir.mkdir(parents=True, exist_ok=True)
        assert_not_legacy_training_dir(dest_dir)
        index = ContentIndex.load(dest_dir / "content_index.json")
        current = count_valid_images(dest_dir)
        needed = args.target_per_class - current
        if needed <= 0:
            print(f"[{brand}] Already at {current}/{args.target_per_class} valid images. Skipping.")
            continue

        models = df[df["brand"] == brand]["model"].dropna().unique().tolist()
        refs = df[df["brand"] == brand]["ref"].dropna().unique().tolist()
        queries = build_queries(brand, models, refs)
        print(f"[{brand}] current={current} target={args.target_per_class} queries={len(queries)}")

        for query in queries:
            current = count_valid_images(dest_dir)
            if current >= args.target_per_class:
                break
            if query in completed:
                continue
            fetch_count = min(args.images_per_query, args.target_per_class - current)
            success = False
            try:
                _crawl_bing(query, temp_dir, fetch_count)
                added = ingest_temp_dir(
                    temp_dir,
                    dest_dir=dest_dir,
                    artifact_root=artifact_root,
                    brand=brand,
                    query=query,
                    stats=stats,
                    index=index,
                )
                print(f"  query={query!r} stored=+{added} total={count_valid_images(dest_dir)}")
                success = True
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("query failed %r: %s", query, exc)
                success = False
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            mark_query_completed(state, query, success=success)
            state["stats"] = stats.as_dict()
            save_resume_state(state_path, state)
            time.sleep(1)

        final = count_valid_images(dest_dir)
        shortfall = max(0, args.target_per_class - final)
        print(f"[{brand}] final={final}/{args.target_per_class} shortfall={shortfall}")

    print(f"Done in {(time.time() - start) / 60:.1f} minutes. stats={stats.as_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
